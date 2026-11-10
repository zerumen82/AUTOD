# Validación visual — Modo Calidad 1080p y 4K (2026-11-10)

## Qué se valida

El acabado real de `run_top_tier_quality` (v5.1322) a lado mayor HD (1920) y 4K
(3840), con y sin restauración facial. **Este protocolo no declara PASS**: los
contratos CPU y el script sólo aportan medidas y recortes; la aceptación visual
la firma el usuario.

## Antes de ejecutar

1. Reabrir AutoAuto: la UI abierta con v5.1320 no carga este código.
2. No lanzar la validación mientras haya un render o una edición en curso.
3. Comprobar GPU libre: el 4K usa ESRGAN x2 real y puede tardar; 8 GB es el
   entorno de referencia. El 8K sigue apagado (`allow_8k=false`).
4. Elegir dos fotos reales por tamaño:
   - un retrato con cara grande (identidad, halo, piel),
   - una escena con paredes/cielo/piel amplia (bandas, textura, rejilla).
   Evitar contenido explícito para esta revisión.

## Ejecución

```bash
# Plan sin ejecutar: tier efectivo y topes por tamaño de la foto
venv/Scripts/python.exe diagnostics/probe_quality_visual_1080_4k_20261110.py --dry-run --photo <foto.jpg>

# HD y 4K con caras
venv/Scripts/python.exe diagnostics/probe_quality_visual_1080_4k_20261110.py --photo <foto.jpg> --tier both

# Sin restauración facial (para separar el efecto del acabado clásico)
venv/Scripts/python.exe diagnostics/probe_quality_visual_1080_4k_20261110.py --photo <foto.jpg> --tier 4k --no-faces
```

Evidencia en `diagnostics/out/quality_visual_20261110/`:
`input.png`, `<tier>_<efectivo>.png`, tres recortes 512 al 100 % por ejecución y
`report.json` con tamaño, nitidez, bandas, recorte, croma, identidad, tiempos y
la nota del acabado.

## Criterios de revisión (marcar en `visual_verdict`)

| criterio | PASS | FAIL |
|---|---|---|
| Bandas y bloques | sin escalones nuevos en cielo/pared/piel | bandas o cuadrícula visibles |
| Halo u óvalos | transición pelo/frente/mandíbula sin línea | óvalo suave o doble borde |
| Identidad facial | cara reconocible, sin piel de plástico | rasgos cambiados o lavados |
| Textura artificial | detalle coherente con la foto | grano inventado, patrón repetido |
| Nitidez y contraste | mejora sin recorte de altas luces/sombras | halos de enfoque o tonos quemados |
| Resolución final | lado mayor 1920 en HD; 4K hasta 3840 según topes | tamaño distinto al anunciado sin nota |

Además, comparar el recorte al 100 % del mismo punto antes/después: el Modo
Calidad no debe cambiar composición, encuadre ni posición de personas.

## Qué hacer con el resultado

- Si algo falla, anotar foto, tier, criterio y recorte; adjuntar la ruta del
  `report.json`. No cambiar umbrales sin esa evidencia.
- Si el 4K agota VRAM en 8 GB, registrar el error y el tamaño real; el tope
  `max_output_pixels` (12 MP) se ajusta en `config/quality_mode.json`.
- El veredicto se escribe en `report.json` → `visual_verdict` con
  `status: PASS|FAIL`, `reviewed_by` y notas. Sólo entonces se puede decir que
  1080p o 4K están validados.
