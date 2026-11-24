# Image Editor — el calvo editaba a otra persona (2026-11-24, v5.1444)

## Petición

> «revisa image editor..porque estaba tardando mucho y no obedeciendo…»

Caso concretado por el usuario: **el último de los calvos**. La queja de
obediencia es específica: **editó a otra persona de la foto**. El motor que
quiere mantener es **LongCat Turbo**, con margen para investigar calidad.
Trabajo autorizado **sin reiniciar la UI**.

## Evidencia de logs

Sesión `logs/ui_start_20261124_100708.log` (v5.1443, 10:07, cerrada a las 10:16):

```
[ImgEditor] Solicitud recibida prompt='el hombre debe ser calvo' engine=auto ... categoria=features
[ImgEditor] Auto: LongCat Turbo como ruta general ...
[ImgEditor] Edad detectada: 49 años (2 cara(s))
[CLIPSeg] Máscara OK para 'a photo of Man': cobertura=16.1%
[CLIPSeg] Máscara OK para 'a photo of el hombre': cobertura=16.0%
[ImgEditor] Sujeto 1/1 medido, cobertura=23.0%: Man must be bald.
[CLIPSeg] Máscara VACÍA para 'calvo' (cobertura=0.00%) - devuelvo None
[CLIPSeg] Máscara OK para 'bald': cobertura=1.4%
[ImgEditor] Zona pedida 'bald' medida (1.6%, 0 px fuera de la silueta)
[ImgEditor] El pelo de la misma cabeza entra en la zona (+14168 px);
            la cara bajo la línea de ojos y las caras ajenas quedan fuera
[ImgEditor] Región medida a 3480x2901 con la foto alrededor (17.6% de máscara)
[ImgEditor] Apariencia: una pasada, 20 pasos, fuerza 1.00
[ComfyUI] Codificando texto+imagen (Qwen VL) | transcurrido 5:14 | sin eventos WS
[ComfyUI] forrtl: error (200): program aborting due to window-CLOSE event
```

La sesión `logs/ui_start_20261124_090042.log` repite el mismo patrón con
«EL HOMBRE DEBE SER CALVO»: 2 caras, sujeto del 31–33 %, y salidas
`img_editor/edit_1795507788.png`, `...7964.png`, `...8111.png`, `...8339.png`.

## Causa raíz

1. `_measured_subject_passes()` une las máscaras CLIPSeg «Man»/«el hombre» en
   **un solo grupo** (`_group_measured_subjects`), sin separar a las dos personas.
2. Como «bald/calvo» no está en la foto, cae al fallback
   `El predicado no tiene zona; entra la región ya medida y se pega entera`.
3. `_head_zone_for_subject()` hacía:
   ```python
   head_union = self._head_mask_from_faces(image, faces)  # TODAS las caras
   zone = head_union & subject_mask                       # sujeto de 2 personas
   ```
   El agujero de generación y el pegado incluían la cabeza ajena.
4. El camino correcto ya existía (`_appearance_head_pass`: mide con
   `masker.probability` qué cara ilumina la frase y exige 1,25× sobre la rival),
   pero solo se invocaba cuando **no** había máscara de sujeto. Aquí sí la había.
5. El stall de Qwen VL venía del recorte 3480×2901: el codificador procesa la
   imagen de trabajo antes de muestrear, aunque el cliente luego baje a ~1 MP.

## Corrección (v5.1444)

**Fase 1 — selección de una sola persona** (`roop/img_editor/img_editor_manager.py`)

- `_head_zone_for_subject()` acepta `raw_prompt`/`translated_prompt` y elige
  **una** cabeza: `_appearance_head_pass` → `_leading_face` sobre la máscara →
  desempate por `det_score` si la máscara es plana. Sin listas de palabras,
  géneros ni tipos de edición escritos a mano.
- Resta las cabezas de las demás caras (`_head_mask_from_faces` por cara) antes
  de enviar la zona al sampler y al pegado.
- Traza: `Cabeza del sujeto medida para la edición de apariencia (frase medida);
  las demás cabezas quedan fuera`.
- Los dos call sites (fallback `deliver_region` y comprobación «parte dentro de
  la cabeza») pasan los prompts.

**Fase 2 — calidad con Turbo, sin cambiar de motor**

- `config/editor_selection.json` → `edit`:
  - `local_appearance_min_steps`: **28** (antes 20 fijos en código).
  - `local_appearance_max_work_side`: **1536** px de lado largo antes de ComfyUI.
  - `_doc_local_appearance_quality` explica ambos valores y su origen medido.
- Aplicado en las **dos** rutas de generación de apariencia local; la segunda
  tenía `appearance_sample` fijo en 20 y el primer parche no la alcanzaba.
- El pegado ya reescala el generado al recorte original con compensación de
  nitidez (`_integrate_local_generation`), así que acotar no cambia la zona.

## Verificación ejecutada

Sin ComfyUI real, sin GPU y sin reiniciar la UI:

| Probe | Resultado |
|---|---|
| `diagnostics/probe_editor_head_zone_single_person_20261124.py` | **PASS** — unión antigua: 38 396 px de la otra cara dentro; arreglo: **0 px**, 33 418 px de la elegida conservados, caso de una sola cara intacto |
| `diagnostics/probe_editor_local_appearance_quality_20261124.py` | **PASS** — config 28/1536; 3480×2901 → 1536×1280; cobertura 14,1449 % → 14,1450 %; pegado 40 000 px dentro y 0 fuera |
| `diagnostics/probe_editor_local_appearance_downscale_path_20261124.py` | **PASS** — ruta real con tope forzado 200: primer pase 200 px, 28 pasos en los 3 `generate` |
| `diagnostics/probe_editor_bald_backhair_20261016.py` | **PASS** — `steps=28`, traza de cabeza única, sin regresión de anclaje |

Comprobaciones: `py_compile`, JSON válido, `git diff --check`, Ruff sin errores
nuevos (quedan 3 F821 `_LONGCAT_FULL_VERSION` presentes en HEAD).
`probe_editor_versatility_integrity_20261105.py` sigue con 2 FAIL de
`RemovalContracts`/LaMa, preexistentes y ajenos a la función tocada.

## Límites — qué NO está cerrado

- **Sin PASS visual.** Todo lo anterior es evidencia técnica de despacho,
  píxel y ruta; no sustituye la revisión de una foto real.
- P3 del documento de pendientes sigue abierto: no se decidieron tope de
  reintentos ni fallback directo.
- La ruta `calidad=True` (TopTier) no se tocó: MEMORY v5.1439 documenta piel
  de cera y tampoco tiene PASS visual.

## Reproducción del PASS visual pendiente

1. Reiniciar la app (el código Python se carga al arrancar).
2. Abrir Image Editor con una foto de **dos personas**, selector `features`,
   prompt «el hombre debe ser calvo», motor `auto` (LongCat Turbo).
3. Confirmar en el log:
   `Cabeza del sujeto medida … (frase medida); las demás cabezas quedan fuera`,
   `Apariencia: una pasada, 28 pasos`, y `Imagen de trabajo acotada` si el
   recorte supera 1536 px.
4. Revisar el PNG guardado: solo el hombre elegido queda calvo; la otra persona,
   el fondo y el encuadre quedan como en el original. Anotar PASS/FAIL aquí.
