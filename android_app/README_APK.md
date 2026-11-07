# AutoAuto — App Android nativa

La aplicación actual usa una interfaz Kotlin (`MainActivity`) y un cliente HTTP
(`GradioClient`) para comunicarse con AutoAuto en el PC. Tiene modos **Editar**
y **Generate**; el procesamiento se ejecuta en el PC. No es una WebView ni
contiene todas las funciones de la interfaz de escritorio.

## Conexión y uso

1. Inicia AutoAuto en el PC y deja accesible su servidor Gradio en la red del
   teléfono. El puerto habitual del proyecto es 7861.
2. En la app, introduce la URL del servidor en el diálogo de conexión. Se
   guarda en las preferencias y se reutiliza en siguientes aperturas. La app
   no escanea automáticamente toda la red local.
3. En **Editar**, elige una imagen de la galería, escribe la instrucción y
   ajusta los controles disponibles. En **Generate**, basta una instrucción.
4. La app muestra progreso y resultado; permite guardar, compartir y reutilizar
   la imagen. Una conexión correcta no garantiza la calidad visual del modelo.

Kontext es el motor predeterminado del Editor. Los alias LongCat antiguos se
normalizan a Kontext en el servidor. El selector visual de regiones del
escritorio tiene endpoints propios; la APK utiliza el contrato de edición
legacy, sin enviar una selección explícita de regiones.

## Contratos con el servidor

`GradioClient` consulta `/config` para descubrir el prefijo de API (normalmente
`/gradio_api`) y `/info` bajo ese prefijo para localizar los endpoints:

- Edición: prefiere `/edit_options` con nueve argumentos; conserva `/edit`
  con ocho para servidores anteriores. El argumento adicional controla la
  preservación de caras en Modo Calidad.
- Generate: `/generate_image`, con diez argumentos. Los controles adicionales
  del evento de escritorio `/generate_image_ui` no se envían desde la APK.
- La imagen se sube mediante `/upload`; las llamadas y el progreso/resultados
  usan `/call/<endpoint>` y `/call/<endpoint>/<event_id>` bajo el prefijo de API.

El cliente conserva los píxeles originales en la subida para Modo Calidad;
las ediciones normales pueden reducir la entrada a un lado máximo de 1024 px.

## Compilar con las herramientas locales

Se requieren Windows, JDK 17, Gradle 8.7 y Android SDK 34 con build-tools 34.0.0,
además de las dependencias Gradle del proyecto. En este workspace las herramientas
están en `android_build_tools/`. Android mínimo: 8.0 (API 26).

Desde la raíz del repositorio, este comando usa las herramientas existentes y
el caché local; no descarga dependencias:

```powershell
$env:JAVA_HOME = (Resolve-Path 'android_app/android_build_tools/jdk17').Path
& 'android_app/android_build_tools/gradle-8.7/bin/gradle.bat' `
  -p 'android_app/autoapp' --offline --no-daemon :app:assembleDebug :app:lintDebug
```

`autoapp/local.properties` debe apuntar al SDK instalado. Si falta una herramienta
o una dependencia en caché, detenerse y solicitar autorización antes de instalar
o descargar. El script legacy `build_apk.bat` puede descargar herramientas y
dependencias; su ejecución requiere esa autorización. Para release exige una
firma configurada por el usuario: no genera claves ni contraseñas predeterminadas.

Salidas habituales:

- Debug: `autoapp/app/build/outputs/apk/debug/app-debug.apk`.
- Release: `autoapp/app/build/outputs/apk/release/app-release.apk`, firmado si
  existe una configuración de firma válida en `autoapp/release-keystore.properties`.

No publicar ni sobrescribir claves de firma. Una compilación correcta no
equivale a instalar y probar la app en un teléfono. Para instalar un APK local,
Android puede pedir permiso para la aplicación desde la que se abre el archivo.

`release-keystore.properties` contiene `storeFile`, `storePassword`, `keyAlias`
y `keyPassword`; debe referenciar una clave existente y permanecer fuera de Git.
Las claves locales previas se conservan para mantener la compatibilidad de las
actualizaciones. Este cambio no rota ni certifica sus credenciales anteriores.

## Archivos principales

- `autoapp/app/src/main/java/com/autoauto/app/MainActivity.kt`: controles,
  preferencias, conexión, galería y presentación de resultados.
- `autoapp/app/src/main/java/com/autoauto/app/GradioClient.kt`: subida de imagen,
  contratos API, progreso y descarga del resultado.
- `autoapp/app/src/main/res/layout/activity_main.xml`: interfaz nativa.
- `autoapp/app/build.gradle.kts`: SDK, versión, dependencias y firma opcional.
- `autoapp/app/src/main/AndroidManifest.xml`: permisos y `FileProvider`.

## Red y permisos

La conexión local admite HTTP sin TLS (`usesCleartextTraffic=true`). El diálogo
de servidor permite introducir una URL accesible desde el teléfono; para acceso
remoto se necesita conectividad existente hacia el PC, sin recompilar una URL
en el código. El manifiesto declara red/WiFi y permisos de imágenes o
almacenamiento según versión de Android; no declara permiso de ubicación.

La validación real de carga, edición, Generate, progreso y guardado en un teléfono
sigue abierta en [validación pendiente](../spec/05-VALIDACION-PENDIENTE.md)
hasta registrar evidencia.
