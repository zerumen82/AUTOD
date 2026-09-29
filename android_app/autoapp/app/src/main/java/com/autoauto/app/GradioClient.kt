package com.autoauto.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import android.util.Log
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID

class GradioClient(private var baseUrl: String) {

    private val SESSION = UUID.randomUUID().toString().replace("-", "").substring(0, 12)
    private var apiPrefix = "/gradio_api"

    /** Callback for real-time progress updates during generation */
    interface ProgressListener {
        fun onProgress(phase: String, percent: Int, detail: String)
    }

    init {
        // Detect api_prefix from /config
        try {
            val conn = openGet("$baseUrl/config")
            if (conn.responseCode == 200) {
                val json = JSONObject(conn.inputStream.bufferedReader().use { it.readText() })
                conn.disconnect()
                val prefix = json.optString("api_prefix", "")
                if (prefix.isNotBlank()) apiPrefix = prefix
            } else conn.disconnect()
        } catch (_: Exception) {}
    }

    // ------------------------------------------------------------------
    //  Test connection
    // ------------------------------------------------------------------
    fun testConnection(): Boolean {
        return try {
            val conn = openGet("$baseUrl/config")
            val ok = conn.responseCode == 200
            conn.disconnect()
            ok
        } catch (_: Exception) { false }
    }

    // ------------------------------------------------------------------
    //  Discover API endpoint name from /gradio_api/info
    // ------------------------------------------------------------------
    fun discoverApiName(): String {
        return try {
            val conn = openGet("$baseUrl${apiPrefix}/info")
            if (conn.responseCode != 200) return "edit"
            val json = JSONObject(conn.inputStream.bufferedReader().use { it.readText() })
            conn.disconnect()
            val named = json.optJSONObject("named_endpoints") ?: return "edit"
            if (named.has("/edit_options")) return "edit_options"
            // Older servers expose the original eight-argument endpoint.
            if (named.has("/edit")) return "edit"
            // Otherwise first named endpoint
            named.keys().forEach { return it.removePrefix("/") }
            "edit"
        } catch (_: Exception) { "edit" }
    }

    // ------------------------------------------------------------------
    //  Upload image -> returns server file path
    // ------------------------------------------------------------------
    fun uploadImage(bitmap: Bitmap, preserveOriginal: Boolean = false, fileName: String = "input.jpg"): String {
        // Quality mode needs the original pixels. Downscaling every upload to
        // 1024px made HD/4K/8K restoration start from a degraded copy.
        val maxDim = maxOf(bitmap.width, bitmap.height)
        val uploadBmp = if (!preserveOriginal && maxDim > 1024) {
            val scale = 1024f / maxDim
            Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true)
        } else bitmap

        val baos = ByteArrayOutputStream()
        val format = if (preserveOriginal) Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG
        uploadBmp.compress(format, if (preserveOriginal) 100 else 92, baos)
        val bytes = baos.toByteArray()

        val boundary = "----AutoAuto${System.currentTimeMillis()}"
        val crlf = "\r\n"
        val twoHyphens = "--"

        val conn = openPost("$baseUrl${apiPrefix}/upload")
        conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")

        conn.outputStream.use { os ->
            os.write("$twoHyphens$boundary$crlf".toByteArray())
        val uploadName = if (preserveOriginal) "input.png" else fileName
        os.write("Content-Disposition: form-data; name=\"files\"; filename=\"$uploadName\"$crlf".toByteArray())
        os.write("Content-Type: ${if (preserveOriginal) "image/png" else "image/jpeg"}$crlf$crlf".toByteArray())
            os.write(bytes)
            os.write("$crlf$twoHyphens$boundary$twoHyphens$crlf".toByteArray())
            os.flush()
        }

        val code = conn.responseCode
        if (code != 200) {
            val err = conn.errorStream?.bufferedReader()?.use { it.readText() } ?: "HTTP $code"
            conn.disconnect()
            throw RuntimeException("Upload failed ($code): $err")
        }

        val response = conn.inputStream.bufferedReader().use { it.readText() }
        conn.disconnect()
        val arr = JSONArray(response)
        return arr.getString(0)
    }

    // ------------------------------------------------------------------
    //  Call API (Gradio 6.x: /gradio_api/call/{endpoint})
    //  Returns event_id, then poll for result
    // ------------------------------------------------------------------
    fun callEdit(
        apiName: String,
        imagePath: String,
        prompt: String,
        engine: String,
        denoise: Float,
        autoregressive: Boolean,
        autoIterations: Int,
        qualityMode: Boolean,
        enhanceTier: String,
        qualityPreserveFaces: Boolean = true,
        progressListener: ProgressListener? = null
    ): EditResult {
        val imgObj = JSONObject().apply {
            put("path", imagePath)
            put("url", "$baseUrl${apiPrefix}/file=$imagePath")
            put("orig_path", imagePath)
        }

        val data = JSONArray().apply {
            put(imgObj)
            put(prompt)
            put(engine)
            put(denoise.toDouble())
            put(autoregressive)
            put(autoIterations)
            put(qualityMode)
            put(enhanceTier)
            if (apiName == "edit_options") put(qualityPreserveFaces)
        }

        val body = JSONObject().apply {
            put("data", data)
        }

        // Step 1: POST to /gradio_api/call/{endpoint} -> get event_id
        val conn = openPost("$baseUrl${apiPrefix}/call/$apiName")
        conn.setRequestProperty("Content-Type", "application/json")

        conn.outputStream.use { os ->
            os.write(body.toString().toByteArray())
            os.flush()
        }

        val code = conn.responseCode
        if (code != 200) {
            val err = conn.errorStream?.bufferedReader()?.use { it.readText() } ?: "HTTP $code"
            conn.disconnect()
            throw RuntimeException("API error ($code): $err")
        }

        val responseText = conn.inputStream.bufferedReader().use { it.readText() }
        conn.disconnect()

        val respJson = JSONObject(responseText)
        val eventId = respJson.optString("event_id", "")
        if (eventId.isBlank()) {
            // Some endpoints return data directly
            return parseResponse(responseText)
        }

        // Step 2: Poll /gradio_api/call/{endpoint}/{event_id} for result
        return pollForResult(apiName, eventId, progressListener)
    }

    // ------------------------------------------------------------------
    //  Poll for result (Gradio 6.x SSE stream)
    // ------------------------------------------------------------------
    private fun pollForResult(apiName: String, eventId: String, progressListener: ProgressListener? = null): EditResult {
        val pollUrl = "$baseUrl${apiPrefix}/call/$apiName/$eventId"
        var attempts = 0
        var consecutiveErrors = 0
        while (attempts < 600) {
            Thread.sleep(1000)
            attempts++
            try {
                val conn = openGet(pollUrl)
                conn.setRequestProperty("Accept", "text/event-stream")
                val code = conn.responseCode
                if (code == 404) {
                    conn.disconnect()
                    return EditResult(null, "Sesion no encontrada o finalizada (404)", null)
                }
                if (code != 200) {
                    conn.disconnect()
                    consecutiveErrors++
                    if (consecutiveErrors >= 10) {
                        return EditResult(null, "Servidor no responde (ComfyUI puede haber crasheado)", null)
                    }
                    Thread.sleep(1000)
                    continue
                }
                consecutiveErrors = 0

                // v5.1075: Read SSE stream line-by-line instead of reading all at once.
                // readText() blocks until EOF, losing all intermediate progress events.
                var eventType = ""
                var eventData = ""
                var foundComplete = false
                var completeData = ""

                try {
                    val reader = conn.inputStream.bufferedReader()
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        val l = line ?: continue
                        when {
                            l.startsWith("event: ") -> eventType = l.removePrefix("event: ").trim()
                            l.startsWith("data: ") -> eventData = l.removePrefix("data: ").trim()
                            l.isEmpty() -> {
                                // Empty line = end of SSE event, process it now
                                when (eventType) {
                                    "complete" -> {
                                        Log.d("GradioClient", "SSE complete, data length=${eventData.length}")
                                        progressListener?.onProgress("Descargando resultado", 95, "")
                                        foundComplete = true
                                        completeData = eventData
                                    }
                                    "error" -> {
                                        conn.disconnect()
                                        return EditResult(null, eventData, null)
                                    }
                                    "heartbeat" -> { /* keep reading */ }
                                    "progress" -> {
                                        try {
                                            val progJson = JSONObject(eventData)
                                            val pct = (progJson.optDouble("progress", 0.0) * 100).toInt()
                                            val step = progJson.optInt("current_step", 0)
                                            val total = progJson.optInt("total_steps", 0)
                                            val phase = progJson.optString("phase", "Generando")
                                            val detail = if (step > 0 && total > 0) "Paso $step/$total" else ""
                                            progressListener?.onProgress(phase, pct.coerceIn(10, 95), detail)
                                        } catch (_: Exception) {}
                                    }
                                    else -> {
                                        if (eventData.isNotBlank() && eventData != "[DONE]" && !foundComplete) {
                                            Log.d("GradioClient", "SSE event=$eventType, data length=${eventData.length}")
                                            val result = parseEventData(eventData)
                                            if (result != null) { conn.disconnect(); return result }
                                        }
                                    }
                                }
                                // Reset for next event
                                eventType = ""
                                eventData = ""
                            }
                        }
                    }
                } catch (_: Exception) {}
                conn.disconnect()

                // Process complete event after stream ends
                if (foundComplete && completeData.isNotBlank()) {
                    val result = parseEventData(completeData)
                    if (result != null) return result
                    return EditResult(null, "Empty result", null)
                }
            } catch (e: Exception) {
                consecutiveErrors++
                if (consecutiveErrors >= 10) {
                    return EditResult(null, "Servidor no responde (${e.message?.take(50)})", null)
                }
                Thread.sleep(1000)
            }
        }
        return EditResult(null, "Timeout", null)
    }

    // ------------------------------------------------------------------
    //  Download file from server
    // ------------------------------------------------------------------
    fun downloadFile(path: String, progressListener: ProgressListener? = null): ByteArray {
        val url = if (path.startsWith("http")) path else "$baseUrl${apiPrefix}/file=$path"
        progressListener?.onProgress("Descargando", 90, "")
        val conn = openGet(url)
        if (conn.responseCode != 200) throw RuntimeException("Download failed: HTTP ${conn.responseCode}")
        val totalBytes = conn.contentLength
        val bytes = conn.inputStream.use { input ->
            if (totalBytes > 0) {
                val buf = ByteArrayOutputStream(totalBytes)
                val tmp = ByteArray(8192)
                var read: Int
                var downloaded = 0
                while (input.read(tmp).also { read = it } != -1) {
                    buf.write(tmp, 0, read)
                    downloaded += read
                    val pct = (downloaded * 100 / totalBytes).coerceIn(0, 99)
                    progressListener?.onProgress("Descargando", pct, "${downloaded / 1024}KB / ${totalBytes / 1024}KB")
                }
                buf.toByteArray()
            } else {
                input.readBytes()
            }
        }
        conn.disconnect()
        progressListener?.onProgress("Descargando", 100, "")
        return bytes
    }

    // ------------------------------------------------------------------
    //  Parse SSE event data (handles both JSONObject and JSONArray formats)
    // ------------------------------------------------------------------
    private fun parseEventData(eventData: String): EditResult? {
        if (eventData.isBlank() || eventData == "[DONE]") return null
        // Try JSONArray first (Gradio 6.x format: [{"path":...}, "status", null])
        try {
            val arr = JSONArray(eventData)
            if (arr.length() > 0) {
                Log.d("GradioClient", "Parsed result as JSONArray (${arr.length()} items)")
                return extractResult(arr)
            }
        } catch (_: Exception) {}
        // Try JSONObject (Gradio 5.x format: {"output": {"data": [...]}} or {"data": [...]})
        try {
            val json = JSONObject(eventData)
            val data = extractDataArray(json)
            if (data != null) {
                Log.d("GradioClient", "Parsed result as JSONObject (${data.length()} items)")
                return extractResult(data)
            }
        } catch (_: Exception) {}
        return null
    }

    // ------------------------------------------------------------------
    //  Response parsing (fallback for direct JSON responses)
    // ------------------------------------------------------------------
    private fun parseResponse(text: String): EditResult {
        // SSE format: lines starting with "data: "
        if (text.contains("data: ")) {
            var lastData: String? = null
            for (line in text.lines()) {
                if (line.startsWith("data: ")) lastData = line.removePrefix("data: ").trim()
            }
            if (lastData != null) return parseResponse(lastData)
        }

        // Try JSONArray first (Gradio 6.x direct format)
        try {
            val arr = JSONArray(text)
            if (arr.length() > 0) return extractResult(arr)
        } catch (_: Exception) {}

        // Try JSONObject
        val json = try { JSONObject(text) } catch (_: Exception) {
            return EditResult(null, "Parse error", null)
        }

        // Queue response
        if (json.has("hash")) return EditResult(null, "Queued...", null)

        // Direct result
        val data = json.optJSONArray("data") ?: return EditResult(null, "Sin datos", null)
        return extractResult(data)
    }

    private fun extractDataArray(json: JSONObject): JSONArray? {
        // Try output.data (Gradio 6.x format: {"output": {"data": [...]}})
        val output = json.optJSONObject("output")
        if (output != null) {
            val data = output.optJSONArray("data")
            if (data != null) return data
        }
        // Try direct data
        return json.optJSONArray("data")
    }

    private fun extractResult(data: JSONArray): EditResult {
        var image: Bitmap? = null
        var status = ""
        var mask: Bitmap? = null
        for (i in 0 until data.length()) {
            val item = data.opt(i) ?: continue
            when (i) {
                0 -> {
                    image = toBitmap(item)
                    Log.d("GradioClient", "Image item[$i]: type=${item?.javaClass?.simpleName}, result=${image != null} (${image?.width}x${image?.height})")
                }
                1 -> status = item?.toString() ?: ""
                2 -> {
                    mask = toBitmap(item)
                    Log.d("GradioClient", "Mask item[$i]: type=${item?.javaClass?.simpleName}, result=${mask != null}")
                }
            }
        }
        return EditResult(image, status, mask)
    }

    private fun toBitmap(item: Any?, progressListener: ProgressListener? = null): Bitmap? {
        return try {
            when (item) {
                is JSONObject -> {
                    val path = item.optString("path", "")
                    val url = item.optString("url", "")
                    Log.d("GradioClient", "toBitmap JSONObject: path=$path, url=$url")
                    val bytes = when {
                        path.isNotBlank() -> downloadFile(path, progressListener)
                        url.isNotBlank() -> downloadFile(url, progressListener)
                        else -> {
                            Log.w("GradioClient", "toBitmap: no path or url in JSONObject")
                            return null
                        }
                    }
                    val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    if (bmp == null) Log.w("GradioClient", "toBitmap: BitmapFactory returned null for ${bytes.size} bytes")
                    bmp
                }
                is String -> {
                    when {
                        item.startsWith("data:image") -> {
                            val b64 = item.substringAfter("base64,")
                            val bytes = Base64.decode(b64, Base64.DEFAULT)
                            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                        }
                        item.isNotBlank() -> {
                            Log.d("GradioClient", "toBitmap String path: $item")
                            val bytes = downloadFile(item, progressListener)
                            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                        }
                        else -> null
                    }
                }
                else -> {
                    Log.w("GradioClient", "toBitmap: unexpected type ${item?.javaClass?.simpleName}")
                    null
                }
            }
        } catch (e: Exception) {
            Log.e("GradioClient", "toBitmap error: ${e.message}", e)
            null
        }
    }

    // ------------------------------------------------------------------
    //  HTTP helpers
    // ------------------------------------------------------------------
    private fun openPost(url: String) = (URL(url).openConnection() as HttpURLConnection).apply {
        requestMethod = "POST"; doOutput = true
        connectTimeout = 15000; readTimeout = 600000
        instanceFollowRedirects = true
        setRequestProperty("User-Agent", "AutoAuto/2.0")
    }

    private fun openGet(url: String) = (URL(url).openConnection() as HttpURLConnection).apply {
        requestMethod = "GET"
        connectTimeout = 10000; readTimeout = 30000
        instanceFollowRedirects = true
        setRequestProperty("User-Agent", "AutoAuto/2.0")
    }

    // ------------------------------------------------------------------
    //  Call Generate API (txt2img, no image input)
    // ------------------------------------------------------------------
    fun callGenerate(
        prompt: String,
        width: Int,
        height: Int,
        engine: String,
        enhanceQuality: Boolean,
        useRewriter: Boolean,
        progressListener: ProgressListener? = null
    ): EditResult {
        val data = JSONArray().apply {
            put(prompt)
            put(width.toDouble())
            put(height.toDouble())
            put(engine)
            put("photoreal")    // image_style
            put("auto")        // shot_framing
            put("natural")     // lighting
            put("detailed")    // skin_finish
            put(enhanceQuality)
            put(useRewriter)
        }

        val body = JSONObject().apply {
            put("data", data)
        }

        // Discover generate endpoint
        val genApiName = discoverGenerateApi()

        val conn = openPost("$baseUrl${apiPrefix}/call/$genApiName")
        conn.setRequestProperty("Content-Type", "application/json")

        conn.outputStream.use { os ->
            os.write(body.toString().toByteArray())
            os.flush()
        }

        val code = conn.responseCode
        if (code != 200) {
            val err = conn.errorStream?.bufferedReader()?.use { it.readText() } ?: "HTTP $code"
            conn.disconnect()
            throw RuntimeException("API error ($code): $err")
        }

        val responseText = conn.inputStream.bufferedReader().use { it.readText() }
        conn.disconnect()

        val respJson = JSONObject(responseText)
        val eventId = respJson.optString("event_id", "")
        if (eventId.isBlank()) {
            return parseResponse(responseText)
        }

        return pollForResult(genApiName, eventId, progressListener)
    }

    private fun discoverGenerateApi(): String {
        return try {
            val conn = openGet("$baseUrl${apiPrefix}/info")
            if (conn.responseCode != 200) return "generate_image"
            val json = JSONObject(conn.inputStream.bufferedReader().use { it.readText() })
            conn.disconnect()
            val named = json.optJSONObject("named_endpoints") ?: return "generate_image"
            if (named.has("/generate_image")) return "generate_image"
            if (named.has("/generate")) return "generate"
            named.keys().forEach {
                val name = it.removePrefix("/")
                if (name.contains("gen", ignoreCase = true)) return name
            }
            "generate_image"
        } catch (_: Exception) { "generate_image" }
    }

    data class EditResult(val image: Bitmap?, val status: String, val mask: Bitmap?)
}
