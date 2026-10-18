package com.autoauto.app

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.text.InputType
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.Spinner
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
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    // Views
    private lateinit var inputImage: ImageView
    private lateinit var inputPlaceholder: TextView
    private lateinit var inputFrame: View
    private lateinit var promptInput: EditText
    private lateinit var engineSpinner: Spinner
    private lateinit var denoiseSlider: SeekBar
    private lateinit var denoiseValue: TextView
    private lateinit var qualityCheck: CheckBox
    private lateinit var qualityFaceRestoreCheck: CheckBox
    private lateinit var autoCheck: CheckBox
    private lateinit var tierGroup: RadioGroup
    private lateinit var iterationsRow: LinearLayout
    private lateinit var iterationsSlider: SeekBar
    private lateinit var iterationsValue: TextView
    private lateinit var btnTransform: Button
    private lateinit var progressLayout: LinearLayout
    private lateinit var progressText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var progressPercent: TextView
    private lateinit var progressElapsed: TextView
    private lateinit var resultSection: LinearLayout
    private lateinit var tabResult: Button
    private lateinit var tabMask: Button
    private lateinit var resultImage: ImageView
    private lateinit var maskImage: ImageView
    private lateinit var statusMessage: TextView
    private lateinit var btnSave: Button
    private lateinit var btnShare: Button
    private lateinit var btnUseAsInput: Button
    private lateinit var serverStatus: TextView
    private lateinit var statusDot: View
    private lateinit var btnConnect: Button
    private lateinit var galleryRow: LinearLayout
    private lateinit var galleryLabel: TextView
    private lateinit var scrollRoot: android.widget.ScrollView

    // Generate mode views
    private lateinit var tabEdit: Button
    private lateinit var tabGenerate: Button
    private lateinit var generatePanel: LinearLayout
    private lateinit var editSettingsRow: LinearLayout
    private lateinit var editCheckboxesRow: LinearLayout
    private lateinit var btnOrientationH: Button
    private lateinit var btnOrientationV: Button
    private lateinit var genEngineSpinner: Spinner
    private lateinit var genEnhanceCheck: CheckBox
    private lateinit var genRewriterCheck: CheckBox

    // State
    private var selectedBitmap: Bitmap? = null
    private var lastResultBitmap: Bitmap? = null
    private var lastMaskBitmap: Bitmap? = null
    private var lastResultPath: String? = null
    private var client: GradioClient? = null
    private var apiName: String = "edit"
    private var serverUrl: String? = null
    private var isGenerating = false
    private var showingResult = true
    private var isGenerateMode = false
    private var genOrientationH = true
    private var progressStartTime = 0L
    private var restoringSettings = false

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    // Aliases = valores EXACTOS que espera el servidor (img_editor_manager / flux_gen_comfy_client).
    // LongCat retirado (sustituido por Kontext): los alias legacy se normalizan
    // a kontext en el servidor. La APK conserva los valores por compatibilidad
    // con preferencias guardadas; el servidor los resuelve a Kontext.
    // gen: Q4 = default seguro no-NSFW; abliterated/Chroma/SDXL-explicit = NSFW en desplegable.
    private val engines = arrayOf("kontext", "kontext", "klein", "kontext", "kontext", "qwen_edit")
    private val engineLabels = arrayOf(
        "Auto (Kontext)",
        "Kontext (sigue instrucciones)",
        "FLUX.2 Klein 4B (rápido)",
        "Kontext fuerte (legacy LongCat Full)",
        "Kontext (edición fuerte)",
        "Qwen-Image-Edit Q5 (20B)"
    )
    private val genEngines = arrayOf("flux_dev_q4", "flux_dev_abliterated", "schnell", "klein_base")
    private val genEngineLabels = arrayOf(
        "FLUX Dev Q4 (seguro, por defecto)",
        "FLUX Abliterated (NSFW)",
        "FLUX Schnell (rápido)",
        "FLUX.2 Klein 4B (rápido + realista)"
    )
    private var selectedEngine = 0
    private var selectedGenEngine = 0

    // Persistence
    private fun prefs() = getSharedPreferences("autoauto", Context.MODE_PRIVATE)
    private fun getManualUrl(): String? = prefs().getString("server_url", null)
    private fun saveManualUrl(url: String) { prefs().edit().putString("server_url", url).apply() }

    private fun saveSettings() {
        if (restoringSettings) return
        prefs().edit().apply {
            putInt("engine", selectedEngine)
            putInt("denoise", denoiseSlider.progress)
            putBoolean("quality", qualityCheck.isChecked)
            putBoolean("qualityFaceRestore", qualityFaceRestoreCheck.isChecked)
            putBoolean("autoregressive", autoCheck.isChecked)
            putInt("iterations", iterationsSlider.progress)
            putString("tier", when (tierGroup.checkedRadioButtonId) {
                R.id.tier4k -> "4k"; R.id.tier8k -> "8k"; else -> "hd"
            })
            putString("prompt", promptInput.text.toString())
            putString("lastResultPath", lastResultPath)
            putBoolean("isGenerateMode", isGenerateMode)
            putInt("genEngine", selectedGenEngine)
            putString("genEngineAlias", genEngines[selectedGenEngine])
            putBoolean("genOrientationH", genOrientationH)
            putBoolean("genEnhance", genEnhanceCheck.isChecked)
            putBoolean("genRewriter", genRewriterCheck.isChecked)
            apply()
        }
    }

    private fun restoreSettings() {
        restoringSettings = true
        try {
        val p = prefs()
        selectedEngine = p.getInt("engine", 0)
        // v5.1111: la lista de engines cambió (6 opciones ahora) — clamp para
        // evitar ArrayIndexOutOfBounds con preferencias guardadas de la v1.0/1.1.
        if (selectedEngine >= engines.size) selectedEngine = 0
        engineSpinner.setSelection(selectedEngine)
        denoiseSlider.progress = p.getInt("denoise", 0)
        denoiseValue.text = if (denoiseSlider.progress == 0) "Auto" else String.format("%.1f", denoiseSlider.progress / 20f)
        qualityCheck.isChecked = p.getBoolean("quality", false)
        qualityFaceRestoreCheck.isChecked = p.getBoolean("qualityFaceRestore", true)
        autoCheck.isChecked = p.getBoolean("autoregressive", false)
        iterationsSlider.progress = p.getInt("iterations", 0)
        iterationsValue.text = "${iterationsSlider.progress + 1}"
        val tier = p.getString("tier", "hd") ?: "hd"
        when (tier) { "4k" -> tierGroup.check(R.id.tier4k); "8k" -> tierGroup.check(R.id.tier8k); else -> tierGroup.check(R.id.tierHd) }
        promptInput.setText(p.getString("prompt", "") ?: "")
        // Generate mode
        val defaultGenAlias = getString(R.string.default_generation_engine)
        val savedGenAlias = p.getString("genEngineAlias", null)
        // La versión anterior guardaba el índice: 0=Abliterated, 1=Schnell,
        // 2=Klein. El índice 0 era también su default, de modo que se migra
        // al nuevo default seguro; las otras selecciones conservan su motor.
        val legacyGenEngines = arrayOf("flux_dev_abliterated", "schnell", "klein_base")
        val legacyAlias = legacyGenEngines.getOrNull(p.getInt("genEngine", 0))
        val selectedAlias = savedGenAlias ?: when (legacyAlias) {
            "schnell", "klein_base" -> legacyAlias
            else -> defaultGenAlias
        }
        selectedGenEngine = genEngines.indexOf(selectedAlias).takeIf { it >= 0 }
            ?: genEngines.indexOf(defaultGenAlias).takeIf { it >= 0 }
            ?: 0
        genEngineSpinner.setSelection(selectedGenEngine)
        genEnhanceCheck.isChecked = p.getBoolean("genEnhance", false)
        genRewriterCheck.isChecked = p.getBoolean("genRewriter", false)
        if (p.getBoolean("isGenerateMode", false)) switchToGenerateMode() else switchToEditMode()
        } finally {
            restoringSettings = false
        }
    }

    private fun restoreLastResult() {
        val path = prefs().getString("lastResultPath", null) ?: return
        scope.launch {
            val bmp = try {
                if (path.startsWith("content://")) {
                    contentResolver.openInputStream(Uri.parse(path))?.use { BitmapFactory.decodeStream(it) }
                } else {
                    BitmapFactory.decodeFile(path)
                }
            } catch (_: Exception) { null }
            if (bmp != null && !isFinishing) withContext(Dispatchers.Main) {
                lastResultBitmap = bmp
                lastResultPath = path
                resultImage.setImageBitmap(bmp)
                resultSection.visibility = View.VISIBLE
                statusMessage.text = "Resultado anterior"
            }
        }
    }

    // Image picker
    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) loadImageFromUri(uri)
    }

    // Runtime permission for gallery
    private val requestPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) loadGallery()
    }

    private fun ensureGalleryPermission() {
        val perm = if (Build.VERSION.SDK_INT >= 33) android.Manifest.permission.READ_MEDIA_IMAGES
                   else android.Manifest.permission.READ_EXTERNAL_STORAGE
        if (ContextCompat.checkSelfPermission(this, perm) != PackageManager.PERMISSION_GRANTED) {
            requestPermission.launch(perm)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Global crash handler — writes to file so user can share without adb
        Thread.setDefaultUncaughtExceptionHandler { _, t ->
            try {
                val sw = StringWriter()
                t.printStackTrace(PrintWriter(sw))
                val log = "CRASH: ${t.message}\n$sw"
                Log.e("AutoAuto", log, t)
                val f = File(getExternalFilesDir(null), "crash.log")
                f.writeText(log)
            } catch (_: Exception) {}
        }

        setContentView(R.layout.activity_main)
        bindViews()
        setupEngineSpinner()
        setupGenEngineSpinner()
        setupDenoiseSlider()
        setupIterationsSlider()
        setupCheckboxes()
        setupTabs()
        setupTabSwitcher()
        setupButtons()
        setupInputImage()
        restoreSettings()
        ensureGalleryPermission()

        if (savedInstanceState != null) {
            savedInstanceState.getByteArray("inputBitmap")?.let {
                selectedBitmap = BitmapFactory.decodeByteArray(it, 0, it.size)
                showInputBitmap()
            }
            serverUrl = savedInstanceState.getString("serverUrl")
        }

        val saved = getManualUrl()
        if (saved != null && client == null) connectTo(saved)
        else if (client == null) {
            showDisconnected()
            scope.launch {
                val found = quickDiscover()
                if (found != null) withContext(Dispatchers.Main) { connectTo(found) }
            }
        }
        loadGallery()
        restoreLastResult()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        serverUrl?.let { outState.putString("serverUrl", it) }
        selectedBitmap?.let {
            // Scale down to max 1024px to avoid exceeding bundle size limit
            val maxDim = maxOf(it.width, it.height)
            val scaled = if (maxDim > 1024) {
                val scale = 1024f / maxDim
                Bitmap.createScaledBitmap(it, (it.width * scale).toInt(), (it.height * scale).toInt(), true)
            } else it
            val baos = ByteArrayOutputStream()
            scaled.compress(Bitmap.CompressFormat.JPEG, 80, baos)
            outState.putByteArray("inputBitmap", baos.toByteArray())
        }
    }

    override fun onStop() { super.onStop(); saveSettings() }

    override fun onDestroy() { scope.cancel(); super.onDestroy() }

    // ---- View binding ----
    private fun bindViews() {
        inputImage = findViewById(R.id.inputImage)
        inputPlaceholder = findViewById(R.id.inputPlaceholder)
        inputFrame = findViewById(R.id.inputImageFrame)
        promptInput = findViewById(R.id.promptInput)
        engineSpinner = findViewById(R.id.engineSpinner)
        denoiseSlider = findViewById(R.id.denoiseSlider)
        denoiseValue = findViewById(R.id.denoiseValue)
        qualityCheck = findViewById(R.id.qualityCheck)
        qualityFaceRestoreCheck = findViewById(R.id.qualityFaceRestoreCheck)
        autoCheck = findViewById(R.id.autoCheck)
        tierGroup = findViewById(R.id.tierGroup)
        iterationsRow = findViewById(R.id.iterationsRow)
        iterationsSlider = findViewById(R.id.iterationsSlider)
        iterationsValue = findViewById(R.id.iterationsValue)
        btnTransform = findViewById(R.id.btnTransform)
        progressLayout = findViewById(R.id.progressLayout)
        progressText = findViewById(R.id.progressText)
        progressBar = findViewById(R.id.progressBar)
        progressPercent = findViewById(R.id.progressPercent)
        progressElapsed = findViewById(R.id.progressElapsed)
        resultSection = findViewById(R.id.resultSection)
        tabResult = findViewById(R.id.tabResult)
        tabMask = findViewById(R.id.tabMask)
        resultImage = findViewById(R.id.resultImage)
        maskImage = findViewById(R.id.maskImage)
        statusMessage = findViewById(R.id.statusMessage)
        btnSave = findViewById(R.id.btnSave)
        btnShare = findViewById(R.id.btnShare)
        btnUseAsInput = findViewById(R.id.btnUseAsInput)
        serverStatus = findViewById(R.id.serverStatus)
        statusDot = findViewById(R.id.statusDot)
        btnConnect = findViewById(R.id.btnConnect)
        galleryRow = findViewById(R.id.galleryRow)
        galleryLabel = findViewById(R.id.galleryLabel)
        scrollRoot = findViewById(R.id.scrollRoot)
        // Generate mode
        tabEdit = findViewById(R.id.tabEdit)
        tabGenerate = findViewById(R.id.tabGenerate)
        generatePanel = findViewById(R.id.generatePanel)
        editSettingsRow = findViewById(R.id.editSettingsRow)
        editCheckboxesRow = findViewById(R.id.editCheckboxesRow)
        btnOrientationH = findViewById(R.id.btnOrientationH)
        btnOrientationV = findViewById(R.id.btnOrientationV)
        genEngineSpinner = findViewById(R.id.genEngineSpinner)
        genEnhanceCheck = findViewById(R.id.genEnhanceCheck)
        genRewriterCheck = findViewById(R.id.genRewriterCheck)
    }

    // ---- Engine spinner ----
    private fun setupEngineSpinner() {
        val adapter = object : ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, engineLabels) {
            override fun getView(position: Int, convertView: View?, parent: android.view.ViewGroup): View {
                val v = super.getView(position, convertView, parent)
                v.findViewById<TextView>(android.R.id.text1).setTextColor(0xFFE2E8F0.toInt())
                return v
            }
            override fun getDropDownView(position: Int, convertView: View?, parent: android.view.ViewGroup): View {
                val v = super.getDropDownView(position, convertView, parent)
                v.setBackgroundColor(0xFF1E293B.toInt())
                v.findViewById<TextView>(android.R.id.text1).setTextColor(0xFFE2E8F0.toInt())
                return v
            }
        }
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        engineSpinner.adapter = adapter
        engineSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) { selectedEngine = pos; saveSettings() }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }
    }

    // ---- Denoise slider (0..20 -> 0.0..1.0, 0=Auto) ----
    private fun setupDenoiseSlider() {
        denoiseSlider.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                denoiseValue.text = if (progress == 0) "Auto" else String.format("%.1f", progress / 20f)
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) { saveSettings() }
        })
    }

    // ---- Iterations slider (1..20, default 3) ----
    private fun setupIterationsSlider() {
        iterationsSlider.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                iterationsValue.text = "${progress + 1}"
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) { saveSettings() }
        })
    }

    // ---- Tab switching (Edit / Generate) ----
    private fun setupTabSwitcher() {
        tabEdit.setOnClickListener { switchToEditMode() }
        tabGenerate.setOnClickListener { switchToGenerateMode() }
        btnOrientationH.setOnClickListener { setOrientation(true) }
        btnOrientationV.setOnClickListener { setOrientation(false) }
    }

    private fun switchToEditMode() {
        isGenerateMode = false
        tabEdit.setBackgroundColor(0xFF06B6D4.toInt())
        tabEdit.setTextColor(0xFFFFFFFF.toInt())
        tabGenerate.setBackgroundColor(0xFF1E293B.toInt())
        tabGenerate.setTextColor(0xFF94A3B8.toInt())
        inputFrame.visibility = View.VISIBLE
        generatePanel.visibility = View.GONE
        editSettingsRow.visibility = View.VISIBLE
        editCheckboxesRow.visibility = View.VISIBLE
        tierGroup.visibility = if (qualityCheck.isChecked) View.VISIBLE else View.GONE
        qualityFaceRestoreCheck.visibility = if (qualityCheck.isChecked) View.VISIBLE else View.GONE
        iterationsRow.visibility = if (autoCheck.isChecked) View.VISIBLE else View.GONE
        iterationsSlider.visibility = if (autoCheck.isChecked) View.VISIBLE else View.GONE
        btnTransform.text = "TRANSFORMAR"
        saveSettings()
    }

    private fun switchToGenerateMode() {
        isGenerateMode = true
        tabGenerate.setBackgroundColor(0xFF06B6D4.toInt())
        tabGenerate.setTextColor(0xFFFFFFFF.toInt())
        tabEdit.setBackgroundColor(0xFF1E293B.toInt())
        tabEdit.setTextColor(0xFF94A3B8.toInt())
        inputFrame.visibility = View.GONE
        generatePanel.visibility = View.VISIBLE
        editSettingsRow.visibility = View.GONE
        editCheckboxesRow.visibility = View.GONE
        tierGroup.visibility = View.GONE
        qualityFaceRestoreCheck.visibility = View.GONE
        iterationsRow.visibility = View.GONE
        iterationsSlider.visibility = View.GONE
        btnTransform.text = "GENERAR"
        saveSettings()
    }

    private fun setOrientation(horizontal: Boolean) {
        genOrientationH = horizontal
        if (horizontal) {
            btnOrientationH.setBackgroundColor(0xFF06B6D4.toInt())
            btnOrientationH.setTextColor(0xFFFFFFFF.toInt())
            btnOrientationV.setBackgroundColor(0xFF1E293B.toInt())
            btnOrientationV.setTextColor(0xFF94A3B8.toInt())
        } else {
            btnOrientationV.setBackgroundColor(0xFF06B6D4.toInt())
            btnOrientationV.setTextColor(0xFFFFFFFF.toInt())
            btnOrientationH.setBackgroundColor(0xFF1E293B.toInt())
            btnOrientationH.setTextColor(0xFF94A3B8.toInt())
        }
        saveSettings()
    }

    // ---- Generate engine spinner ----
    private fun setupGenEngineSpinner() {
        val adapter = object : ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, genEngineLabels) {
            override fun getView(position: Int, convertView: View?, parent: android.view.ViewGroup): View {
                val v = super.getView(position, convertView, parent)
                v.findViewById<TextView>(android.R.id.text1).setTextColor(0xFFE2E8F0.toInt())
                return v
            }
            override fun getDropDownView(position: Int, convertView: View?, parent: android.view.ViewGroup): View {
                val v = super.getDropDownView(position, convertView, parent)
                v.setBackgroundColor(0xFF1E293B.toInt())
                v.findViewById<TextView>(android.R.id.text1).setTextColor(0xFFE2E8F0.toInt())
                return v
            }
        }
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        genEngineSpinner.adapter = adapter
        genEngineSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) { selectedGenEngine = pos; saveSettings() }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }
    }

    // ---- Checkboxes (show/hide tier + iterations) ----
    private fun setupCheckboxes() {
        qualityCheck.setOnCheckedChangeListener { _, checked ->
            tierGroup.visibility = if (checked) View.VISIBLE else View.GONE
            qualityFaceRestoreCheck.visibility = if (checked) View.VISIBLE else View.GONE
            saveSettings()
        }
        qualityFaceRestoreCheck.setOnCheckedChangeListener { _, _ -> saveSettings() }
        autoCheck.setOnCheckedChangeListener { _, checked ->
            val vis = if (checked) View.VISIBLE else View.GONE
            iterationsRow.visibility = vis
            iterationsSlider.visibility = vis
            saveSettings()
        }
        tierGroup.setOnCheckedChangeListener { _, _ -> saveSettings() }
    }

    // ---- Result/Mask tabs ----
    private fun setupTabs() {
        tabResult.setOnClickListener {
            showingResult = true
            resultImage.visibility = View.VISIBLE
            maskImage.visibility = View.GONE
            tabResult.setBackgroundColor(0xFF06B6D4.toInt())
            tabResult.setTextColor(0xFFFFFFFF.toInt())
            tabMask.setBackgroundColor(0xFF1E293B.toInt())
            tabMask.setTextColor(0xFF94A3B8.toInt())
        }
        tabMask.setOnClickListener {
            showingResult = false
            resultImage.visibility = View.GONE
            maskImage.visibility = View.VISIBLE
            tabMask.setBackgroundColor(0xFF06B6D4.toInt())
            tabMask.setTextColor(0xFFFFFFFF.toInt())
            tabResult.setBackgroundColor(0xFF1E293B.toInt())
            tabResult.setTextColor(0xFF94A3B8.toInt())
        }
    }

    // ---- Buttons ----
    private fun setupButtons() {
        inputFrame.setOnClickListener { pickImage.launch("image/*") }

        btnConnect.setOnClickListener {
            val crashFile = File(getExternalFilesDir(null), "crash.log")
            val crashLog = if (crashFile.exists()) crashFile.readText() else null

            val items = mutableListOf("URL del servidor")
            if (crashLog != null) items.add("Ver ultimo crash log")

            AlertDialog.Builder(this)
                .setTitle("AutoAuto")
                .setItems(items.toTypedArray()) { _, which ->
                    when (which) {
                        0 -> showConnectDialog()
                        1 -> {
                            AlertDialog.Builder(this)
                                .setTitle("Crash Log")
                                .setMessage(crashLog ?: "Sin crashes")
                                .setPositiveButton("Copiar") { _, _ ->
                                    val cm = getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                                    cm.setPrimaryClip(android.content.ClipData.newPlainText("crash", crashLog))
                                    Toast.makeText(this, "Copiado", Toast.LENGTH_SHORT).show()
                                }
                                .setNegativeButton("Borrar") { _, _ -> crashFile.delete(); Toast.makeText(this, "Borrado", Toast.LENGTH_SHORT).show() }
                                .setNeutralButton("Cerrar", null)
                                .show()
                        }
                    }
                }
                .show()
        }

        btnTransform.setOnClickListener { startGeneration() }

        btnSave.setOnClickListener { saveResult() }
        btnShare.setOnClickListener { shareResult() }
        btnUseAsInput.setOnClickListener {
            if (lastResultBitmap != null) {
                selectedBitmap = lastResultBitmap
                showInputBitmap()
                // v5.1074: Si estamos en modo Generate, cambiar a Edit automáticamente
                if (isGenerateMode) {
                    switchToEditMode()
                    Toast.makeText(this, "Imagen cargada en Editor", Toast.LENGTH_SHORT).show()
                } else {
                    scrollRoot.smoothScrollTo(0, 0)
                    Toast.makeText(this, "Resultado como entrada", Toast.LENGTH_SHORT).show()
                }
            }
        }

        resultImage.setOnClickListener {
            if (lastResultBitmap != null) showFullscreen(lastResultBitmap!!)
        }
        maskImage.setOnClickListener {
            if (lastMaskBitmap != null) showFullscreen(lastMaskBitmap!!)
        }
    }

    // ---- INPUT IMAGE ----
    private fun showConnectDialog() {
        val input = EditText(this).apply {
            hint = "IP del PC (ej: 192.168.1.50:7861)"
            setText(getManualUrl() ?: "")
            inputType = InputType.TYPE_CLASS_TEXT
            selectAll()
        }
        AlertDialog.Builder(this)
            .setTitle("Servidor AutoAuto")
            .setMessage("URL del servidor Gradio")
            .setView(input)
            .setPositiveButton("Conectar") { _, _ ->
                val txt = input.text.toString().trim()
                if (txt.isNotEmpty()) {
                    val url = if (txt.startsWith("http")) txt.trimEnd('/') else "http://$txt"
                    saveManualUrl(url)
                    connectTo(url)
                }
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }

    // ---- Input image ----
    private fun setupInputImage() { inputImage.scaleType = ImageView.ScaleType.CENTER_CROP }

    private fun loadImageFromUri(uri: Uri) {
        try {
            val input = contentResolver.openInputStream(uri) ?: return
            val bmp = BitmapFactory.decodeStream(input)
            input.close()
            if (bmp != null) { selectedBitmap = bmp; showInputBitmap() }
        } catch (e: Exception) {
            Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showInputBitmap() {
        inputImage.setImageBitmap(selectedBitmap)
        inputPlaceholder.visibility = View.GONE
    }

    // ---- Server connection ----
    private fun connectTo(url: String) {
        serverUrl = url
        showDisconnected()
        scope.launch {
            val c = try { GradioClient(url) } catch (_: Exception) { null }
            val ok = try { c?.testConnection() == true } catch (_: Exception) { false }
            if (ok && c != null) {
                val name = try { c.discoverApiName() } catch (_: Exception) { "edit" }
                withContext(Dispatchers.Main) {
                    client = c
                    apiName = name
                    serverStatus.text = url.removePrefix("http://")
                    statusDot.setBackgroundColor(0xFF22C55E.toInt())
                    btnConnect.text = "OK"
                }
            } else withContext(Dispatchers.Main) { showDisconnected() }
        }
    }

    private fun showDisconnected() {
        serverStatus.text = "Sin conexion"
        statusDot.setBackgroundColor(0xFFEF4444.toInt())
        btnConnect.text = "..."
    }

    // ---- Progress helpers ----
    private fun showProgress(phase: String, percent: Int = 0, detail: String = "") {
        progressStartTime = if (progressStartTime == 0L) System.currentTimeMillis() else progressStartTime
        progressLayout.visibility = View.VISIBLE
        progressBar.progress = percent
        progressPercent.text = "$percent%"
        progressText.text = phase
        val elapsed = (System.currentTimeMillis() - progressStartTime) / 1000
        val timeStr = if (elapsed < 60) "${elapsed}s" else "${elapsed / 60}m ${elapsed % 60}s"
        progressElapsed.text = if (detail.isNotBlank()) "$timeStr - $detail" else timeStr
    }

    private fun hideProgress() {
        progressLayout.visibility = View.GONE
        progressStartTime = 0L
    }

    private fun quickDiscover(): String? {
        val fallback = "http://100.122.235.80:7861"
        return try {
            val conn = URL("$fallback/config").openConnection() as HttpURLConnection
            conn.connectTimeout = 1500; conn.readTimeout = 1500
            if (conn.responseCode == 200) {
                val body = conn.inputStream.bufferedReader().use { it.readText() }; conn.disconnect()
                if (body.contains("\"components\"")) fallback else null
            } else { conn.disconnect(); null }
        } catch (_: Exception) { null }
    }

    // ---- GENERATION ----
    @SuppressLint("SetTextI18n")
    private fun startGeneration() {
        if (isGenerating) return
        val c = client
        if (c == null) { Toast.makeText(this, "Conectate al servidor", Toast.LENGTH_SHORT).show(); return }
        val prompt = promptInput.text.toString().trim()

        if (isGenerateMode) {
            // Generate mode: no image needed
            if (prompt.isEmpty()) {
                Toast.makeText(this, "Escribe un prompt para generar", Toast.LENGTH_SHORT).show(); return
            }
            startGenerateMode(c, prompt)
        } else {
            // Edit mode: image required
            val bmp = selectedBitmap
            if (bmp == null) { Toast.makeText(this, "Selecciona una imagen", Toast.LENGTH_SHORT).show(); return }
            if (prompt.isEmpty() && !qualityCheck.isChecked) {
                Toast.makeText(this, "Escribe un prompt o activa Modo Calidad", Toast.LENGTH_SHORT).show(); return
            }
            startEditMode(c, bmp, prompt)
        }
    }

    @SuppressLint("SetTextI18n")
    private fun startGenerateMode(c: GradioClient, prompt: String) {
        isGenerating = true
        btnTransform.isEnabled = false
        resultSection.visibility = View.GONE
        showProgress("Preparando generacion...", 5, "Prompt: ${prompt.take(30)}...")
        progressLayout.post { scrollRoot.smoothScrollBy(0, progressLayout.top - 50) }

        val width = if (genOrientationH) 1152 else 768
        val height = if (genOrientationH) 768 else 1152

        scope.launch {
            try {
                withContext(Dispatchers.Main) { showProgress("Generando con ${genEngineLabels[selectedGenEngine]}...", 10) }
                val result = c.callGenerate(
                    prompt = prompt,
                    width = width,
                    height = height,
                    engine = genEngines[selectedGenEngine],
                    enhanceQuality = genEnhanceCheck.isChecked,
                    useRewriter = genRewriterCheck.isChecked,
                    progressListener = object : GradioClient.ProgressListener {
                        override fun onProgress(phase: String, percent: Int, detail: String) {
                            scope.launch(Dispatchers.Main) { showProgress(phase, percent.coerceIn(10, 95), detail) }
                        }
                    }
                )
                withContext(Dispatchers.Main) {
                    if (result.image != null) {
                        showProgress("Guardando...", 98)
                        lastResultBitmap = result.image
                        lastMaskBitmap = result.mask
                        resultImage.setImageBitmap(result.image)
                        if (result.mask != null) maskImage.setImageBitmap(result.mask)
                        tabMask.visibility = if (result.mask != null) View.VISIBLE else View.GONE
                        resultSection.visibility = View.VISIBLE
                        statusMessage.text = "Listo - toca 'Usar' para editarlo"
                        lastResultPath = saveBitmapToDownloads(result.image, "gen_${timestamp()}.png")
                        if (lastResultPath == null) Log.w("AutoAuto", "Save gen to Downloads failed")
                        else Log.d("AutoAuto", "Saved gen: $lastResultPath")
                        loadGallery()
                        resultSection.post { scrollRoot.smoothScrollBy(0, resultSection.top - 20) }
                    } else {
                        resultSection.visibility = View.VISIBLE
                        statusMessage.text = result.status.ifBlank { "Error desconocido" }
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    resultSection.visibility = View.VISIBLE
                    statusMessage.text = "Error: ${e.message}"
                }
            } finally {
                withContext(Dispatchers.Main) {
                    isGenerating = false
                    btnTransform.isEnabled = true
                    hideProgress()
                }
            }
        }
    }

    @SuppressLint("SetTextI18n")
    private fun startEditMode(c: GradioClient, bmp: Bitmap, prompt: String) {
        isGenerating = true
        btnTransform.isEnabled = false
        resultSection.visibility = View.GONE
        progressStartTime = 0L

        val denoise = denoiseSlider.progress / 20f
        val iterations = iterationsSlider.progress + 1

        scope.launch {
            try {
                withContext(Dispatchers.Main) { showProgress("Subiendo imagen...", 5) }
                val path = c.uploadImage(bmp, preserveOriginal = qualityCheck.isChecked)
                withContext(Dispatchers.Main) { showProgress("Generando...", 15, "Motor: ${engineLabels[selectedEngine].take(25)}") }
                progressLayout.post { scrollRoot.smoothScrollBy(0, progressLayout.top - 50) }

                val result = c.callEdit(
                    apiName = apiName,
                    imagePath = path,
                    prompt = prompt,
                    engine = engines[selectedEngine],
                    denoise = denoise,
                    autoregressive = autoCheck.isChecked,
                    autoIterations = iterations,
                    qualityMode = qualityCheck.isChecked,
                    qualityPreserveFaces = qualityFaceRestoreCheck.isChecked,
                    enhanceTier = when {
                        findViewById<RadioButton>(R.id.tier4k).isChecked -> "4k"
                        findViewById<RadioButton>(R.id.tier8k).isChecked -> "8k"
                        else -> "hd"
                    },
                    progressListener = object : GradioClient.ProgressListener {
                        override fun onProgress(phase: String, percent: Int, detail: String) {
                            scope.launch(Dispatchers.Main) { showProgress(phase, percent.coerceIn(15, 95), detail) }
                        }
                    }
                )

                withContext(Dispatchers.Main) {
                    if (result.image != null) {
                        showProgress("Guardando...", 98)
                        lastResultBitmap = result.image
                        lastMaskBitmap = result.mask
                        resultImage.setImageBitmap(result.image)
                        if (result.mask != null) maskImage.setImageBitmap(result.mask)
                        tabMask.visibility = if (result.mask != null) View.VISIBLE else View.GONE
                        resultSection.visibility = View.VISIBLE
                        statusMessage.text = result.status.ifBlank { "Listo" }

                        lastResultPath = saveBitmapToDownloads(result.image, "edit_${timestamp()}.png")
                        if (lastResultPath == null) Log.w("AutoAuto", "Save to Downloads failed")
                        else Log.d("AutoAuto", "Saved: $lastResultPath")
                        loadGallery()

                        // Auto-scroll to result
                        resultSection.post {
                            scrollRoot.smoothScrollBy(0, resultSection.top - 20)
                        }
                    } else {
                        resultSection.visibility = View.VISIBLE
                        statusMessage.text = result.status.ifBlank { "Error desconocido" }
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    resultSection.visibility = View.VISIBLE
                    statusMessage.text = "Error: ${e.message}"
                }
            } finally {
                withContext(Dispatchers.Main) {
                    isGenerating = false
                    btnTransform.isEnabled = true
                    hideProgress()
                }
            }
        }
    }

    // ---- SAVE / SHARE ----
    private fun saveResult() {
        val bmp = lastResultBitmap ?: return
        val path = saveBitmapToDownloads(bmp, "edit_${timestamp()}.png")
        if (path != null) {
            lastResultPath = path
            Toast.makeText(this, "Guardado en Fotos/AutoAuto", Toast.LENGTH_LONG).show()
            loadGallery()
        } else Toast.makeText(this, "Error al guardar", Toast.LENGTH_SHORT).show()
    }

    private fun shareResult() {
        val bmp = lastResultBitmap ?: return
        try {
            // v5.1158 — Compartir SIEMPRE desde un FileProvider/cache propio:
            // las URIs de MediaStore pueden fallar en algunas apps destino si el
            // flag FLAG_GRANT no se propaga bien con createChooser. Con el cache
            // file + ClipData el permiso de lectura viaja garantizado.
            val cacheDir = File(cacheDir, "share").apply { mkdirs() }
            val shareFile = File(cacheDir, "autoauto_${timestamp()}.png")
            shareFile.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }

            val uri = androidx.core.content.FileProvider.getUriForFile(
                this, "${packageName}.fileprovider", shareFile
            )
            val send = Intent(Intent.ACTION_SEND).apply {
                type = "image/png"
                putExtra(Intent.EXTRA_STREAM, uri)
                // ClipData = el permiso FLAG_GRANT_READ se propaga a TODAS las apps
                // del chooser (algunos destinos ignoran el extra flags)
                clipData = android.content.ClipData.newRawUri("image", uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(send, "Compartir resultado"))
        } catch (e: Exception) { Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show() }
    }

    private fun saveBitmapToDownloads(bmp: Bitmap, name: String): String? {
        return try {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                // Scoped storage: Pictures/AutoAuto es una colección de imágenes
                // real, visible de inmediato para Galería/Google Photos. Downloads
                // puede guardar el fichero pero muchos fabricantes no lo muestran
                // como foto, que parecía un fallo de guardado.
                val resolver = contentResolver
                val values = android.content.ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, name)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/AutoAuto")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
                val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null
                val ok = try {
                    resolver.openOutputStream(uri, "w")?.use { stream ->
                        bmp.compress(Bitmap.CompressFormat.PNG, 100, stream).also { compressed ->
                            if (compressed) stream.flush()
                        }
                    } ?: false
                } catch (e: Exception) {
                    Log.e("AutoAuto", "No se pudo escribir $uri", e)
                    false
                }
                if (ok) {
                    values.clear()
                    values.put(MediaStore.Images.Media.IS_PENDING, 0)
                    if (resolver.update(uri, values, null, null) != 1) {
                        resolver.delete(uri, null, null)
                        return null
                    }
                } else {
                    resolver.delete(uri, null, null)
                    return null
                }
                uri.toString()
            } else {
                // Legacy storage
                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                if (!dir.exists()) dir.mkdirs()
                val file = File(dir, name)
                file.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
                // v5.1158: notificar a la galería (Android <10 no indexa sin broadcast)
                android.media.MediaScannerConnection.scanFile(this, arrayOf(file.absolutePath), arrayOf("image/png"), null)
                file.absolutePath
            }
        } catch (e: Exception) {
            Log.e("AutoAuto", "Error guardando resultado en Fotos/AutoAuto", e)
            null
        }
    }

    // ---- FULLSCREEN VIEWER ----
    @SuppressLint("ClickableViewAccessibility")
    private fun showFullscreen(bmp: Bitmap) {
        val overlay = android.widget.FrameLayout(this).apply {
            layoutParams = android.widget.FrameLayout.LayoutParams(
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT
            )
            setBackgroundColor(0xDD000000.toInt())
            isClickable = true; isFocusable = true
        }
        overlay.addView(ImageView(this).apply {
            layoutParams = android.widget.FrameLayout.LayoutParams(
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT
            )
            scaleType = ImageView.ScaleType.FIT_CENTER; setImageBitmap(bmp)
        })
        val dp = resources.displayMetrics.density
        // Close button — larger for mobile (56dp)
        overlay.addView(Button(this).apply {
            text = "X"; setTextColor(0xFFFFFFFF.toInt()); setBackgroundColor(0x88000000.toInt())
            textSize = 16f; paint.isFakeBoldText = true
            layoutParams = android.widget.FrameLayout.LayoutParams(
                (56 * dp).toInt(), (56 * dp).toInt()
            ).apply { gravity = android.view.Gravity.TOP or android.view.Gravity.END; topMargin = (16 * dp).toInt(); marginEnd = (16 * dp).toInt() }
        })
        val rootView = findViewById<android.view.ViewGroup>(android.R.id.content)
        rootView.addView(overlay)
        val dismiss = { rootView.removeView(overlay); true }
        (overlay.getChildAt(1) as? Button)?.setOnClickListener { dismiss() }
        overlay.setOnClickListener { dismiss() }
        // Back button dismisses fullscreen
        overlay.setOnKeyListener { _, keyCode, _ ->
            if (keyCode == android.view.KeyEvent.KEYCODE_BACK) { dismiss() } else false
        }
    }

    // ---- GALLERY ----
    private fun loadGallery() {
        galleryRow.removeAllViews()
        val dp = resources.displayMetrics.density
        val size = (100 * dp).toInt()
        val margin = (6 * dp).toInt()

        // Always show the current result first
        val currentBmp = lastResultBitmap
        if (currentBmp != null) {
            galleryRow.visibility = View.VISIBLE
            galleryLabel.visibility = View.VISIBLE
            val iv = ImageView(this).apply {
                layoutParams = LinearLayout.LayoutParams(size, size).apply { marginEnd = margin }
                scaleType = ImageView.ScaleType.CENTER_CROP
                setImageBitmap(currentBmp)
                setBackgroundColor(0xFF1E293B.toInt())
            }
            iv.setOnClickListener { showFullscreen(currentBmp) }
            galleryRow.addView(iv)
        }

        // Then add saved images from Downloads
        val images = findDownloadImages()
        for (path in images.take(20)) {
            val iv = ImageView(this).apply {
                layoutParams = LinearLayout.LayoutParams(size, size).apply { marginEnd = margin }
                scaleType = ImageView.ScaleType.CENTER_CROP
                setBackgroundColor(0xFF1E293B.toInt())
            }
            scope.launch {
                val bmp = try {
                    if (path.startsWith("content://")) {
                        val uri = Uri.parse(path)
                        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                    } else {
                        BitmapFactory.decodeFile(path)
                    }
                } catch (_: Exception) { null }
                withContext(Dispatchers.Main) {
                    if (bmp != null && !isFinishing) {
                        iv.setImageBitmap(bmp)
                        iv.setOnClickListener {
                            scope.launch {
                                val fullBmp = try {
                                    if (path.startsWith("content://")) {
                                        val uri = Uri.parse(path)
                                        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                                    } else {
                                        BitmapFactory.decodeFile(path)
                                    }
                                } catch (_: Exception) { null }
                                withContext(Dispatchers.Main) {
                                    if (fullBmp != null) showFullscreen(fullBmp)
                                }
                            }
                        }
                    }
                }
            }
            galleryRow.addView(iv)
        }

        if (galleryRow.childCount == 0) {
            galleryRow.visibility = View.GONE
            galleryLabel.visibility = View.GONE
        } else {
            galleryRow.visibility = View.VISIBLE
            galleryLabel.visibility = View.VISIBLE
        }
    }

    @SuppressLint("InlinedApi")
    private fun findDownloadImages(): List<String> {
        return try {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                // Scoped storage: query MediaStore
                val images = mutableListOf<String>()
                val projection = arrayOf(
                    MediaStore.Images.Media._ID,
                    MediaStore.Images.Media.DISPLAY_NAME,
                    MediaStore.Images.Media.DATE_MODIFIED
                )
                val selection = "(${MediaStore.Images.Media.DISPLAY_NAME} LIKE ? OR ${MediaStore.Images.Media.DISPLAY_NAME} LIKE ?)"
                val selectionArgs = arrayOf("edit_%.png", "gen_%.png")
                val sortOrder = "${MediaStore.Images.Media.DATE_MODIFIED} DESC"
                contentResolver.query(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    projection, selection, selectionArgs, sortOrder
                )?.use { cursor ->
                    val idCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
                    while (cursor.moveToNext() && images.size < 20) {
                        val id = cursor.getLong(idCol)
                        val uri = android.content.ContentUris.withAppendedId(
                            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id
                        )
                        images.add(uri.toString())
                    }
                }
                images
            } else {
                // Legacy storage
                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                if (!dir.exists()) return emptyList()
                dir.listFiles()
                    ?.filter { it.isFile && (it.name.endsWith(".png", true) || it.name.endsWith(".jpg", true)) && (it.name.startsWith("edit_") || it.name.startsWith("gen_")) }
                    ?.sortedByDescending { it.lastModified() }
                    ?.map { it.absolutePath }
                    ?: emptyList()
            }
        } catch (_: Exception) { emptyList() }
    }

    private fun timestamp(): String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
}
