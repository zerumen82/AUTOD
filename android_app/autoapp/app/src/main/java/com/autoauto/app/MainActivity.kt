package com.autoauto.app

import android.annotation.SuppressLint
import android.app.DownloadManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.text.InputType
import android.util.Base64
import android.view.KeyEvent
import android.view.View
import android.webkit.DownloadListener
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.Inet4Address
import java.net.NetworkInterface
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * AutoAuto Editor — cliente Android (WebView).
 *
 * Al abrir la app: escanea la red local buscando el servidor AutoAuto (puerto 7861)
 * y, en cuanto lo encuentra, carga el ImgEditor a pantalla completa. Si el servidor
 * no está encendido, muestra "servidor no encontrado" y un botón de reintentar.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private lateinit var statusText: TextView
    private lateinit var loadingLayout: LinearLayout
    private lateinit var errorLayout: LinearLayout
    private lateinit var errorText: TextView

    private var serverUrl: String? = null
    private var fileChooserCallback: ValueCallback<Array<Uri>>? = null

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private val pickImage = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val callback = fileChooserCallback
        if (result.resultCode == RESULT_OK && result.data != null) {
            callback?.onReceiveValue(arrayOf(result.data!!.data!!))
        } else {
            callback?.onReceiveValue(null)
        }
        fileChooserCallback = null
    }

    private var pendingDownload: Triple<String, String, String>? = null

    private val requestWritePermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        pendingDownload?.let { (url, userAgent, contentDisposition) ->
            if (granted) {
                enqueueDownload(url, userAgent, contentDisposition)
            } else {
                Toast.makeText(this, "Permiso para guardar denegado", Toast.LENGTH_SHORT).show()
            }
        }
        pendingDownload = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        webView = findViewById(R.id.webview)
        statusText = findViewById(R.id.statusText)
        loadingLayout = findViewById(R.id.loadingLayout)
        errorLayout = findViewById(R.id.errorLayout)
        errorText = findViewById(R.id.errorText)

        setupWebView()

        findViewById<Button>(R.id.btnSettings).setOnClickListener { showServerDialog() }

        findViewById<Button>(R.id.btnRetry).setOnClickListener {
            showLoading(getString(R.string.connecting))
            startDiscovery()
        }

        serverUrl = savedInstanceState?.getString("serverUrl")
        if (serverUrl != null) {
            loadServer(serverUrl!!)
        } else {
            // Si el usuario fijo una IP/URL manual, usarla; si no, escanear la red.
            val manual = getManualUrl()
            if (!manual.isNullOrBlank()) {
                loadServer(manual)
            } else {
                showLoading(getString(R.string.connecting))
                startDiscovery()
            }
        }
    }

    // ---------------------------------------------------------------
    //  Servidor manual (IP/URL recordada) + dialogo de ajustes
    // ---------------------------------------------------------------
    private fun getManualUrl(): String? =
        getSharedPreferences("autoauto", Context.MODE_PRIVATE).getString("server_url", null)

    private fun saveManualUrl(url: String) {
        getSharedPreferences("autoauto", Context.MODE_PRIVATE)
            .edit().putString("server_url", url.trim().trimEnd('/')).apply()
    }

    private fun clearManualUrl() {
        getSharedPreferences("autoauto", Context.MODE_PRIVATE)
            .edit().remove("server_url").apply()
    }

    private fun showServerDialog() {
        val input = EditText(this).apply {
            hint = "IP o URL del servidor"
            setText(getManualUrl() ?: "")
            inputType = InputType.TYPE_CLASS_TEXT
            selectAll()
        }
        val wrap = android.widget.LinearLayout.LayoutParams(
            android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
            android.widget.LinearLayout.LayoutParams.WRAP_CONTENT
        )
        wrap.setMargins(60, 20, 60, 0)
        input.layoutParams = wrap

        AlertDialog.Builder(this)
            .setTitle("Servidor AutoAuto")
            .setMessage("Pon la IP o URL del PC y se guardara automaticamente (ej: 192.168.1.50:7861)")
            .setView(input)
            .setPositiveButton("Guardar y conectar") { _, _ ->
                val txt = input.text.toString().trim()
                if (txt.isNotEmpty()) {
                    saveManualUrl(txt)
                    val url = if (txt.startsWith("http://") || txt.startsWith("https://")) txt.trimEnd('/') else "http://$txt"
                    loadServer(url)
                }
            }
            .setNeutralButton("Auto-descubrir") { _, _ ->
                clearManualUrl()
                showLoading(getString(R.string.connecting))
                startDiscovery()
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }

    // ---------------------------------------------------------------
    //  Descubrimiento automático del servidor
    // ---------------------------------------------------------------
    private fun startDiscovery() {
        scope.launch {
            val found = findServer()
            withContext(Dispatchers.Main) {
                if (!isFinishing && found != null) {
                    serverUrl = found
                    loadServer(found)
                } else if (!isFinishing) {
                    showError(getString(R.string.error_notfound))
                }
            }
        }
    }

    private suspend fun findServer(): String? = withContext(Dispatchers.IO) {
        // Subred actual del dispositivo WLAN + subredes domésticas habituales
        val prefixes = LinkedHashSet<String>()
        currentWifiSubnet()?.let { prefixes.add(it) }
        prefixes.addAll(
            listOf(
                "192.168.0.", "192.168.1.", "192.168.2.", "192.168.3.",
                "192.168.10.", "192.168.100.", "10.0.0.", "10.0.1.",
                "172.16.0.", "172.16.1."
            )
        )

        val port = SERVER_PORT
        val found = AtomicBoolean(false)
        val result = AtomicReference<String>()

        // (Opcional) URL publica fija (p.ej. https://xxx.gradio.live) si server_share=true
        // en config.yaml. Solo se usa si la dejas escrita en FALLBACK_URL (companion object).
        if (FALLBACK_URL.isNotBlank() && probeFull(FALLBACK_URL)) {
            return@withContext FALLBACK_URL
        }

        // Escanear en paralelo; los hosts sin servicio se rechazan al instante.
        val pool = Executors.newFixedThreadPool(64)
        try {
            for (prefix in prefixes) {
                if (found.get()) break
                val futures = (1..254).map { i ->
                    pool.submit<Boolean> {
                        if (found.get()) return@submit false
                        val host = prefix + i
                        if (probe(host, port)) {
                            found.set(true)
                            result.set("http://$host:$port")
                            true
                        } else {
                            false
                        }
                    }
                }
                // Espera acotada por prefijo.
                val deadline = System.currentTimeMillis() + PREFIX_TIMEOUT_MS
                while (System.currentTimeMillis() < deadline && !found.get()) {
                    Thread.sleep(40)
                }
                futures.forEach { f -> f.cancel(true) }
            }
        } finally {
            pool.shutdownNow()
        }
        result.get()
    }

    /** Comprueba si un host sirve el Gradio de AutoAuto (firma en /config). */
    private fun probe(host: String, port: Int): Boolean {
        var conn: HttpURLConnection? = null
        return try {
            val url = URL("http://$host:$port/config")
            conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = CONNECT_TIMEOUT_MS
            conn.readTimeout = READ_TIMEOUT_MS
            conn.requestMethod = "GET"
            conn.instanceFollowRedirects = false
            val code = conn.responseCode
            if (code == 200) {
                val body = conn.inputStream.bufferedReader().use { it.readText() }
                body.contains("\"components\"") || body.contains("\"version\"")
            } else {
                false
            }
        } catch (_: Exception) {
            false
        } finally {
            try { conn?.disconnect() } catch (_: Exception) { }
        }
    }


    /** Comprueba una URL base completa (p.ej. publica gradio.live). */
    private fun probeFull(base: String): Boolean {
        var conn: HttpURLConnection? = null
        return try {
            val url = URL(base.trimEnd('/') + "/config")
            conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = CONNECT_TIMEOUT_MS
            conn.readTimeout = READ_TIMEOUT_MS
            conn.requestMethod = "GET"
            conn.instanceFollowRedirects = false
            val code = conn.responseCode
            if (code == 200) {
                val body = conn.inputStream.bufferedReader().use { it.readText() }
                body.contains("\"components\"") || body.contains("\"version\"")
            } else {
                false
            }
        } catch (_: Exception) {
            false
        } finally {
            try { conn?.disconnect() } catch (_: Exception) { }
        }
    }

    /** Recupera la subred /24 de la interfaz de red activa (WLAN). */
    private fun currentWifiSubnet(): String? {
        try {
            val nis = NetworkInterface.getNetworkInterfaces() ?: return null
            for (ni in nis) {
                if (!ni.isUp || ni.isLoopback || ni.isVirtual) continue
                val addrs = ni.inetAddresses ?: continue
                for (ia in addrs) {
                    if (ia is Inet4Address && !ia.isLoopbackAddress) {
                        val parts = ia.hostAddress.split(".")
                        if (parts.size == 4) {
                            return "${parts[0]}.${parts[1]}.${parts[2]}."
                        }
                    }
                }
            }
        } catch (_: Exception) { }
        return null
    }

    // ---------------------------------------------------------------
    //  WebView del ImgEditor
    // ---------------------------------------------------------------
    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        val s = webView.settings
        s.javaScriptEnabled = true
        s.domStorageEnabled = true
        s.databaseEnabled = true
        s.loadWithOverviewMode = true
        s.useWideViewPort = true
        s.allowFileAccess = true
        s.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
        s.mediaPlaybackRequiresUserGesture = false

        webView.webViewClient = object : WebViewClient() {
            override fun onPageStarted(view: WebView?, url: String?, favicon: Bitmap?) {
                showLoading("Conectando con $url…")
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                if (!isFinishing) {
                    loadingLayout.visibility = View.GONE
                    errorLayout.visibility = View.GONE
                    webView.visibility = View.VISIBLE
                }
            }

            @Deprecated("Deprecated in Java")
            override fun onReceivedError(
                view: WebView?, errorCode: Int, description: String?, failingUrl: String?
            ) {
                if (failingUrl == serverUrl) {
                    showError("No se pudo cargar el servidor:\n$description")
                }
            }

            override fun onReceivedError(
                view: WebView?, request: WebResourceRequest?, error: WebResourceError?
            ) {
                super.onReceivedError(view, request, error)
                if (request?.isForMainFrame == true) {
                    showError("No se pudo cargar el servidor:\n${error?.description}")
                }
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onShowFileChooser(
                webView: WebView?,
                filePathCallback: ValueCallback<Array<Uri>>?,
                fileChooserParams: FileChooserParams?
            ): Boolean {
                fileChooserCallback = filePathCallback
                val intent = fileChooserParams?.createIntent()
                    ?: Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }
                try {
                    pickImage.launch(intent)
                } catch (e: Exception) {
                    filePathCallback?.onReceiveValue(null)
                    fileChooserCallback = null
                }
                return true
            }
        }

        // Descargar imágenes/cuando la UI ofrece botón de descarga (ej. gradio).
        webView.setDownloadListener(DownloadListener { url, userAgent, contentDisposition, mimetype, _ ->
            handleDownload(url, userAgent, contentDisposition, mimetype)
        })

        // Volver atrás en el historial del WebView en lugar de salir de la app.
        webView.setOnKeyListener { _, keyCode, _ ->
            if (keyCode == KeyEvent.KEYCODE_BACK && webView.canGoBack() && webView.visibility == View.VISIBLE) {
                webView.goBack()
                true
            } else {
                false
            }
        }
    }

    // ---------------------------------------------------------------
    //  Descarga de imágenes (resultados del editor)
    // ---------------------------------------------------------------
    private fun handleDownload(url: String, userAgent: String?, contentDisposition: String?, mimetype: String?) {
        val ua = userAgent ?: webView.settings.userAgentString ?: "Mozilla/5.0"
        val cd = contentDisposition ?: ""
        // Android <10 necesita permiso de escritura para guardar en Downloads.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q &&
            ContextCompat.checkSelfPermission(this, android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED
        ) {
            pendingDownload = Triple(url, ua, cd)
            requestWritePermission.launch(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
        } else {
            downloadDirectly(url, ua, cd)
        }
    }

    /**
     * Descarga directa con las cookies del WebView. El DownloadManager de Android
     * abre su propio cliente HTTP que NO reenvía la cookie de sesión de Gradio
     * (`_gradio_session`), y Gradio rechaza `/file` sin ella -> Android muestra
     * "rechazado por el servidor". Este método navega con la cookie real + Referer.
     */
    private fun downloadDirectly(url: String, userAgent: String, contentDisposition: String) {
        scope.launch {
            val result = runCatching {
                val conn = (URL(url).openConnection() as HttpURLConnection)
                conn.connectTimeout = 20000
                conn.readTimeout = 60000
                conn.requestMethod = "GET"
                conn.instanceFollowRedirects = true
                conn.setRequestProperty("User-Agent", userAgent)
                val cookies = android.webkit.CookieManager.getInstance().getCookie(url)
                if (!cookies.isNullOrBlank()) conn.setRequestProperty("Cookie", cookies)
                serverUrl?.let { conn.setRequestProperty("Referer", it.trimEnd('/') + "/") }
                val code = conn.responseCode
                if (code != 200) {
                    return@runCatching "HTTP $code"
                }
                conn.setRequestProperty("Accept", "*/*")
                val stream = conn.inputStream
                val buffer = java.io.ByteArrayOutputStream()
                stream.copyTo(buffer)
                stream.close()
                conn.disconnect()
                val fileName = parseFileName(url, contentDisposition)
                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                if (!dir.exists()) dir.mkdirs()
                val file = java.io.File(dir, fileName)
                java.io.FileOutputStream(file).use { it.write(buffer.toByteArray()) }
                file.absolutePath
            }
            withContext(Dispatchers.Main) {
                result.fold(
                    onSuccess = { path ->
                        if (path.startsWith("HTTP")) {
                            Toast.makeText(this@MainActivity, "El servidor rechazo la descarga: $path", Toast.LENGTH_LONG).show()
                        } else {
                            Toast.makeText(this@MainActivity, "Descargado: $path", Toast.LENGTH_LONG).show()
                        }
                    },
                    onFailure = { e ->
                        Toast.makeText(this@MainActivity, "Error al descargar: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                )
            }
        }
    }

    @Suppress("DEPRECATION")
    private fun enqueueDownload(url: String, userAgent: String, contentDisposition: String) {
        try {
            val fileName = parseFileName(url, contentDisposition)
            val request = DownloadManager.Request(Uri.parse(url))
                .setTitle(fileName)
                .setDescription("Descarga de AutoAuto")
                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                .setAllowedOverMetered(true)
                .setAllowedOverRoaming(true)
            request.addRequestHeader("User-Agent", userAgent)
            val cookies = android.webkit.CookieManager.getInstance().getCookie(url)
            if (!cookies.isNullOrBlank()) request.addRequestHeader("Cookie", cookies)
            serverUrl?.let { request.addRequestHeader("Referer", it.trimEnd('/') + "/") }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                request.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, fileName)
            } else {
                @Suppress("DEPRECATION")
                request.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, fileName)
            }
            val dm = getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
            dm.enqueue(request)
            Toast.makeText(this, "Descargando: $fileName", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Error al descargar: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    /** Extrae nombre de archivo del Content-Disposition o de la URL. */
    private fun parseFileName(url: String, contentDisposition: String): String {
        val fallback = url.substringAfterLast('/').substringBefore('?').ifBlank { "resultado.jpg" }
        val match = Regex("filename\\*?=(?:UTF-8'')?\"?([^\";]+)\"?").find(contentDisposition)
        val raw = match?.groupValues?.get(1) ?: fallback
        val clean = raw.replace(Regex("[\\\\/:*?\"<>|]"), "_").trim()
        return clean.ifBlank { "resultado.jpg" }
    }

    private fun loadServer(url: String) {
        serverUrl = url
        webView.visibility = View.VISIBLE
        loadingLayout.visibility = View.VISIBLE
        errorLayout.visibility = View.GONE
        webView.loadUrl(url)
    }

    private fun showLoading(msg: String) {
        loadingLayout.visibility = View.VISIBLE
        errorLayout.visibility = View.GONE
        statusText.text = msg
    }

    private fun showError(msg: String) {
        loadingLayout.visibility = View.GONE
        webView.visibility = View.GONE
        errorLayout.visibility = View.VISIBLE
        errorText.text = msg
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        serverUrl?.let { outState.putString("serverUrl", it) }
        webView.saveState(outState)
    }

    override fun onRestoreInstanceState(savedInstanceState: Bundle) {
        super.onRestoreInstanceState(savedInstanceState)
        webView.restoreState(savedInstanceState)
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
        webView.destroy()
    }

    companion object {
        private const val SERVER_PORT = 7861
        private const val CONNECT_TIMEOUT_MS = 450
        private const val READ_TIMEOUT_MS = 450
        private const val PREFIX_TIMEOUT_MS = 2800
        // Opcional: URL publica fija (server_share=true o Tailscale) que se comprueba
        // antes del escaneo local. Dejala en "" para no usarla.
        // Tailscale: IP estable 100.x.x.x del PC (funciona desde cualquier red con la app Tailscale).
        private const val FALLBACK_URL: String = "http://100.122.235.80:7861"
    }
}

