# AutoAuto

Aplicación local para generar, editar, animar y mejorar imágenes, además de sustituir caras en fotos y vídeos. La interfaz de escritorio reúne los flujos en una sola ventana y usa la GPU del PC.

**Estado:** v5.1397 (18 de noviembre de 2026), en desarrollo. Las funciones están disponibles, pero varias categorías de calidad visual siguen en validación. Consulta los [pendientes actuales](docs/PENDIENTES_2026-11-13.md) antes de interpretar un probe técnico como resultado visual aprobado.

## Características

| Área | Qué permite hacer |
| --- | --- |
| **Generate** | Crear imágenes desde texto con los motores instalados, estilos, iluminación, encuadre, LoRAs compatibles y hasta cuatro resultados por solicitud. |
| **Image Editor** | Editar una foto con instrucciones en lenguaje natural: apariencia, objetos, personas, entorno y cámara. Incluye selección visual de regiones y comprobaciones para limitar cambios fuera de la zona solicitada. |
| **Face Swap** | Sustituir caras en fotos o vídeos usando fotos de origen, selección de caras, seguimiento, máscaras faciales y mejora de detalle. |
| **Animate Image** | Crear vídeo a partir de una imagen. Integra LTX-2 con audio nativo, además de rutas FramePack y Wan según el motor y los recursos disponibles. |
| **Mejorar fotos** | Comparar antes/después, recorrer las fotos importadas, aplicar 15 presets de fábrica o guardar presets propios, ajustar controles por foto y exportar un lote. |
| **Android** | App nativa para **Editar** y **Generate** conectada al servidor del PC por la red local. El procesamiento sigue ocurriendo en el PC. |

## Funcionamiento

- **Procesamiento local:** la UI usa Gradio y pywebview; Generate, Image Editor y Animate comparten una instancia local de ComfyUI. El enlace público de Gradio está desactivado por defecto, aunque la configuración actual acepta conexiones desde la red local.
- **Entorno probado:** Windows, Python 3.10, NVIDIA RTX 3060 Ti de 8 GB y CUDA 12.4. Algunos motores requieren más memoria o descargan trabajo a la CPU.
- **Modelos y entorno virtual:** `models/`, `venv/` y la instalación de ComfyUI no forman parte del repositorio. Clonar el código por sí solo no prepara los modelos ni las dependencias.

### Arranque con el entorno ya preparado

Desde la raíz del proyecto en PowerShell:

```powershell
& .\venv\Scripts\python.exe run.py
```

El lanzador `AutoAuto.exe`, cuando está disponible, ejecuta ese mismo `run.py` mediante `venv\Scripts\python.exe`; necesita el entorno y los modelos junto al proyecto. La UI se abre localmente en `http://127.0.0.1:7861`; con `server_name: 0.0.0.0` también acepta conexiones de la red local. ComfyUI usa el puerto `8188`.

## Documentación del proyecto

- [Principios y criterios de validación](spec/00-PRINCIPIOS.md)
- [Arquitectura y módulos](spec/01-ARQUITECTURA.md)
- [Requisitos y trabajo pendiente](spec/02-REQUISITOS.md)
- [Deudas y límites conocidos](spec/03-PENDIENTES.md)
- [App Android](android_app/README_APK.md)
- [Bitácora de cambios y pruebas](MEMORY.md)

El proyecto conserva resultados y pruebas diagnósticas en `diagnostics/`. Una prueba de API o de píxeles no sustituye la revisión visual de una foto o un vídeo real.
