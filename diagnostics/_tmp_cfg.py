import json, pathlib
p = pathlib.Path("config/editor_selection.json")
data = json.loads(p.read_text(encoding="utf-8"))
edit = data["edit"]
edit["_doc_residual"] = (
    "v5.1365 (P4): residual_failure_fraction 0,40 -> 0,15. Medido el "
    "2026-11-15 con fotos reales y el mismo segmentador del editor: una "
    "eliminacion CORRECTA (zapatos, 2026-11-07) deja 1,5 % de residuo y un "
    "caso donde el objeto sigue visible (casco con LaMa, 2026-11-14) mide "
    "91,8 %. Con 40 % el umbral daba por bueno un objeto que seguia a la "
    "vista; 15 % es 10x el residuo real de una eliminacion correcta y "
    "6x por debajo del caso en que el objeto sigue presente. "
    "clothing_residual_failure_fraction: residual de ROPA, otro criterio "
    "(el estado lo decide la verificacion por piel); se mueve de codigo a "
    "configuracion sin cambiar su valor."
)
edit["residual_failure_fraction"] = 0.15
edit["clothing_residual_failure_fraction"] = 0.4
p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("ok", edit["residual_failure_fraction"], edit["clothing_residual_failure_fraction"])
