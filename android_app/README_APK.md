# AutoAuto Editor — App Android (APK)

App Android **WebView** que se conecta **sola** al servidor AutoAuto de tu PC y permite
editar imágenes exactamente como en **ImgEditor** (misma UI de Gradio del PC).

## Qué hace la app

- **Cero configuración**: No escribes IPs ni URLs. Abres la app y listo.
- **Auto-descubrimiento**: Al abrirse, escanea la red local (puerto **7861**, el de Gradio)
  y, en cuanto encuentra el PC con AutoAuto **encendido**, carga el ImgEditor a pantalla
  completa.
- **Servidor manual (opcional)**: botón **⚙** (arriba a la derecha) para pegar una IP o URL
  concreta una sola vez. La app **la recuerda** y la usa en siguientes aperturas; también
  tiene "Auto-descubrir" para volver al escaneo automático. Útil si el PC está en otra
  subred, por VPN o con URL pública.
- **Si el server está apagado**: muestra *"No se encontró el servidor AutoAuto encendido"*
  con un botón **Reintentar** (no crashea).
- **Editar imágenes**: Como usa la UI de Gradio del PC, tienes el ImgEditor completo
  (subir foto desde la galería/cámara del móvil, prompt, motor LongCat/FLUX/Klein, Modo
  Calidad, autoregresivo, etc.).

## Requisitos en el PC

- Windows con Java o sin él: el script descarga un **JDK 17** portátil.
- iOS no: es Android (`.apk`).
- El móvil y el PC deben estar en la **misma red WiFi**.

## Cómo generar el APK (1 clic)

1. Ten AutoAuto instalado/compilado (para el resto de funciones del PC).
2. Ejecuta (doble clic):

   ```
   android_app\build_apk.bat
   ```

3. La primera vez descarga ~2.5 GB de herramientas (JDK 17 + Android SDK + Gradle),
   genera automáticamente una **clave de firma** y compila un APK **release firmado**.
   Cuando termine tu APK estará en:

   ```
   android_app\autoapp\app\build\outputs\apk\release\app-release.apk
   ```

4. Copia este `.apk` a tu Android (USB, Drive, WhatsApp, etc.) e instálalo. En Android
   acepta "Instalar desde esta fuente" si te lo pide (es normal en la primera instalación
   de cualquier app que no viene de Google Play).

## Uso en el móvil

1. Asegúrate de que AutoAuto esté iniciado en el PC (server en puerto 7861).
2. Abre la app en el móvil.
3. Espera a que "busque" el servidor (unos segundos) y se cargue el ImgEditor.
4. Sube una foto, escribe el cambio y pulsa TRANSFORMAR.

## Acceso remoto (fuera de casa) — opcional

Si en `config.yaml` pones `server_share: true`, el PC genera una URL pública tipo
`https://xxxx.gradio.live`. Para que la app la use:

1. Abre `app\src\main\java\com\autoauto\app\MainActivity.kt`.
2. En el `companion object`, escribe tu URL en `FALLBACK_URL`:
   ```kotlin
   private const val FALLBACK_URL: String = "https://tu-nombre.gradio.live"
   ```
3. Vuelve a ejecutar `build_apk.bat`.

## Estructura relevante

- `MainActivity.kt` — WebView + auto-descubrimiento del servidor.
- `activity_main.xml` — pantallas de búsqueda / error / editor.
- `build_apk.bat` — instala toolchain y compila el APK en 1 comando.
- `app/build.gradle.kts` — versión SDK, paquetes y dependencias.

## Notas

- La app necesita permiso de red local; conecta HTTP plano (cleartext) porque el server
  Gradio es local (sin TLS). No requiere permisos de ubicación.
- El `build_apk.bat` genera una **clave de firma** (`release-keystore.jks`) en la primera
  compilación y la reutiliza siempre: así las actualizaciones se instalan encima sin
  problemas. Guarda ese archivo (no lo borres) — es tu firma de la app.
- Es un APK firmado con tu propia clave, listo para instalación directa en cualquier
  Android. Solo para publicar en Google Play necesitarías configurar la firma específica
  de la cuenta, pero para uso personal/instalación directa funciona tal cual.
