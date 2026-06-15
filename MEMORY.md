# 🧠 MEMORIA DEL PROYECTO - AUTOAUTO

## 📅 Última Actualización: 10 de Junio de 2026

### 🚀 Logros Recientes
1.  **Eliminación del Halo Borroso:**
    *   Se refinó la lógica de máscaras en `roop/ProcessMgr.py`.
    *   Reducción de erosión de 3 a 1 píxel y de GaussianBlur de 5 a 3 en el contenido del warp.
    *   Implementación de un `edge_band_align` más fino (3x3) para transiciones nítidas.
    *   Optimización del feathering dinámico según la resolución de la cara.

2.  **Mejoras de Tracking y Calidad (v5.6.9):**
    *   **ADN Maestro:** Mezcla 60/40 entre embedding maestro y actual para mantener identidad en giros bruscos.
    *   **Smoothing Adaptativo:** `f_center` y `kps_ema` ahora responden dinámicamente a la velocidad del movimiento.
    *   **Ghost Tracking:** Mejorada la inercia (EMA 0.70) para proyectar la cara cuando se pierde brevemente la visibilidad.
    *   **M-EMA:** Estabilización de la matriz afín para eliminar el "jitter" o temblor en los bordes.

3.  **Protección de Oclusiones y Boca (v5.5):**
    *   **Oclusión Avanzada:** Nueva detección de "objetos extraños" mediante diferencia LAB (>65) y detalle laplaciano optimizado.
    *   **Mouth-Object Protection:** La preservación de boca ahora es inteligente; si detecta un objeto (comida, micro, mano), aumenta el `m_blend` (hasta 0.65) para proteger el objeto y no "swapear" sobre él.
    *   **Skin-LAB Sensitivity:** Umbral reducido (30 -> 25) para captar manos y dedos con mayor precisión sobre el rostro.

### 📌 Pendiente Inmediato (Prioridad UI/UX)
*   **Selección por Click:** Eliminar sliders y permitir pinchar directamente en la cara de la galería o el video.
*   **Navegación de Frames:** Implementar selector de frame en el tab de FaceSwap para elegir la cara destino de cualquier punto del video.
*   **Refactorización:** `faceswap_tab.py` necesita ser dividido; su tamaño actual dificulta la implementación de estas mejoras de UI.

### 🛠️ Archivos Clave
*   `roop/ProcessMgr.py`: Cerebro del procesamiento y tracking (Actualizado con v5.6.9).
*   `ui/tabs/faceswap_tab.py`: (Próximo objetivo) Interfaz de usuario de FaceSwap.
*   `MEJORAS_UI_CHECKLIST.md`: Seguimiento detallado de peticiones de usuario.
