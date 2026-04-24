# PLAN DE REESTRUCTURACIÓN DE generate_intelligent()

# ESTRUCTURA ACTUAL (incorrecta):
# 1. Secciones de motores (flux, qwen, zimage, hart, omnigen2) → NO usan análisis, always img2img
# 2. FLUJO ESTÁNDAR (SD inpainting) → sí usa análisis, pero solo para inpainting con máscara
# Problema: cuando detecta pose, va al flujo estándar (inpainting) en lugar de img2img completo

# ESTRUCTURA NUEVA (correcta):
# 1. REDIMENSIONAR imagen (código actual 825-855)
# 2. ANÁLISIS de prompt (rewrite, detección pose/cuerpo) → define is_full_generation
# 3. DECISIÓN DE MOTOR:
#    - Si engine指定ado por usuario, usarlo
#    - Si is_full_generation y engine no es "sd" (estándar), usar OmniGen2 (más seguro 8GB)
# 4. GENERACIÓN:
#    - Para OmniGen2/HART/Qwen/FLUX/ZImage: generar img2img completo (sin máscara)
#    - Para SD estándar: inpainting con máscara (solo áreas específicas)
# 5. FACE SWAP / RESTORE si corresponde

# CAMBIOS:

# A. Mover ANÁLISIS (líneas 1217-1287) al inicio, antes de todos los motores
# B. Después del análisis, check: if is_full_generation and engine == "sd": engine = "omnigen2"
# C. Para cada motor, si el prompt original incluye pose/cuerpo_completo, usar denoise bajo (0.6-0.7)
# D. ELIMINAR la sección de inpainting (máscara) para motores diferentes a "sd"
