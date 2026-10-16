# -*- coding: utf-8 -*-
"""v5.1207 — contrato del piso de recarga turbo-aware (CPU, sin servidor).

Fija con datos de producción (edición "el hombre debe ser calvo", 2026-10-13,
RTX 3060 Ti 8 GB) cuándo load() de FluxEditComfyClient descarga el modelo
antes de reutilizarlo:

1. Reutilizar MISMA versión Turbo con >= 4000 MB libres: SIN unload. La
   pasada 2 corrió con 5198 MB libres a 3,1 s/paso sin thrash y la recarga
   limpia cuesta ~164 s hasta el primer paso. Turbo con CFG=1 omite el encode
   negativo, así que no se reproduce el ping-pong TE↔UNet medido antes.
2. Reutilizar MISMA versión Turbo con < 4000 MB libres: unload sincróno.
3. Reutilizar MISMA versión Full: piso 5600 MB (su thrash con encode negativo
   sigue sin desmentir).
4. PRIMERA carga del cliente con VRAM ajena de otra sesión: piso conservador
   5600 MB aunque sea Turbo (no se sabe qué modelo dejó la sesión anterior).
5. No-LongCat o GPU de más de 10 GB: piso histórico 2200 MB, sin cambio.

Ejecutar: venv\\Scripts\\python.exe -m diagnostics.probe_editor_reuse_floor_turbo
"""
from unittest.mock import MagicMock, patch

from roop.img_editor.flux_edit_comfy_client import (
    FluxEditComfyClient,
    _reuse_floor_mb_for,
)

TURBO = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
FULL = "LongCat-Image-Edit-Q4_K_S.gguf"
KLEIN = "flux2-klein-4b-Q4_K_S.gguf"

PASSED = []


def _ok(label):
    PASSED.append(label)
    print(f"PASS {len(PASSED):02d} — {label}")


def _run_load(version, free, loaded=True, queue_idle=True):
    """load() con servidor simulado; devuelve (ok, msg, unload_calls)."""
    client = FluxEditComfyClient()
    if loaded:
        client._loaded = True
        client._flux_version = version
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"UnetLoaderGGUF": {}}
    with patch.object(FluxEditComfyClient, "is_available", return_value=True), \
            patch("roop.img_editor.flux_edit_comfy_client.requests.get",
                  return_value=response), \
            patch("roop.img_editor.flux_edit_comfy_client._server_vram_free_mb",
                  return_value=(free, 8192)), \
            patch("roop.img_editor.flux_edit_comfy_client._queue_is_idle",
                  return_value=queue_idle), \
            patch("roop.img_editor.flux_edit_comfy_client._sync_unload",
                  return_value=True) as unload:
        try:
            ok, msg = client.load(version)
        except Exception as exc:  # rama de primera carga puede seguir y fallar
            ok, msg = False, f"{type(exc).__name__}: {exc}"
    return ok, msg, unload.call_count


def main():
    # 1) Función pura: pisos por versión y contexto.
    assert _reuse_floor_mb_for(TURBO, 8192) == 4000
    assert _reuse_floor_mb_for(FULL, 8192) == 5600
    assert _reuse_floor_mb_for(TURBO, 8192, reuse=False) == 5600
    assert _reuse_floor_mb_for(FULL, 8192, reuse=False) == 5600
    assert _reuse_floor_mb_for(KLEIN, 8192) == 2200
    assert _reuse_floor_mb_for(TURBO, None) == 2200
    assert _reuse_floor_mb_for(TURBO, 16000) == 2200
    _ok("pisos: Turbo reuso 4000, Full 5600, primera carga 5600, no-LongCat/GRAN 2200")

    # 2) Reutilización con VRAM medida en producción (5198 libres).
    ok, msg, unload = _run_load(TURBO, 5198)
    assert ok and "reutilizado" in msg and unload == 0, (msg, unload)
    _ok("Turbo reutilizando con 5198 libres: SIN unload (dato real de pasada 2)")

    # 3) Turbo justo en el filo de 4000 → sí unload; a 4001 → reutiliza.
    ok, _, unload = _run_load(TURBO, 3999)
    assert unload == 1, unload
    ok, msg, unload = _run_load(TURBO, 4001)
    assert ok and unload == 0, (msg, unload)
    _ok("Turbo: <4000 libres descarga, >=4000 reutiliza")

    # 4) Full conserva el piso 5600: 5198 libres → unload; 5700 → reutiliza.
    ok, _, unload = _run_load(FULL, 5198)
    assert unload == 1, unload
    ok, msg, unload = _run_load(FULL, 5700)
    assert ok and unload == 0, (msg, unload)
    _ok("Full (encode negativo): piso 5600 intacto — 5198 descarga, 5700 reutiliza")

    # 5) Primera carga con VRAM ajena: piso conservador 5600 aunque Turbo.
    _, _, unload = _run_load(TURBO, 5249, loaded=False)
    assert unload == 1, unload
    _, _, unload = _run_load(TURBO, 5700, loaded=False)
    assert unload == 0, unload
    _ok("primera carga con VRAM de otra sesión: piso conservador 5600")

    # 6) Sin datos de VRAM (system_stats caído) o cola ocupada: sin unload.
    ok, msg, unload = _run_load(TURBO, None)
    assert ok and unload == 0, (msg, unload)
    ok, _, unload = _run_load(TURBO, 3999, queue_idle=False)
    assert unload == 0, unload
    _ok("VRAM desconocida o cola ocupada: sin unload (comportamiento previo)")

    print(f"\nRESUMEN: {len(PASSED)}/{len(PASSED)} PASS — piso turbo-aware v5.1207")


if __name__ == "__main__":
    main()
