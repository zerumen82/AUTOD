"""Cierre de sesión: al cerrar la webview con la X se libera la GPU y se cierra ComfyUI.

Pedido del usuario (2026-10-20): «cuando se cierra la webview desde la X debe
liberar la gpu siempre y cerrar comfuy».

Contratos que verifica:
  1. La política vive en ``config/session_shutdown.json`` y es válida.
  2. El cierre es idempotente: la llamada de la ventana y la de ``atexit`` no
     duplican trabajo (se cuentan ejecuciones reales de las dos acciones).
  3. Una configuración inválida se rechaza en voz alta y NO ejecuta acciones.
  4. La observación de estado mide VRAM por ``nvidia-smi`` y ve ComfyUI real.
  5. ``roop/core.py`` está cableado y ya no promete conservar ComfyUI.
  6. **Real end-to-end, tres rutas**: la ventana que arrancó ComfyUI
     (propietario), una ventana secundaria que reutiliza una sesión ajena
     (``comfy_process`` limpio) y la salida del proceso por ``atexit``. Las tres
     tienen que acabar con ComfyUI muerto: puerto cerrado, informe JSON.
  7. **VRAM real**: FaceAnalysis se carga en la GPU del proceso principal y se
     comprueba con ``nvidia-smi`` que la descarga devuelve la memoria.

El test real deja ComfyUI detenido, que es el estado deseado tras cerrar la
ventana. Si hay una UI de Gradio sirviendo o la cola de ComfyUI está ocupada,
el probe se niega a interrumpirla.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import roop.session_shutdown as ss  # noqa: E402

OUT = ROOT / "diagnostics" / "out" / "session_shutdown_20261020"
COMFY_PORT = 8188
GRADIO_PORTS = range(7861, 7871)

_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS {name}" + (f" — {detail}" if detail else ""), flush=True)
        return
    message = f"FAIL {name}" + (f" — {detail}" if detail else "")
    print(message, flush=True)
    _failures.append(message)


def reset() -> None:
    ss._done = False
    ss._report = {}


def port_open(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.5)
            return sock.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def listening_pids(port: int) -> list[str]:
    result = subprocess.run(
        f"netstat -ano | findstr :{port}", shell=True, capture_output=True,
        text=True, timeout=15,
    )
    pids = []
    for line in result.stdout.splitlines():
        if "LISTENING" in line:
            parts = line.split()
            if parts and parts[-1].isdigit():
                pids.append(parts[-1])
    return sorted(set(pids))


def gpu_apps() -> dict:
    """{pid: MiB} de los procesos con contexto de cómputo en la GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
    except Exception:
        return {}
    apps = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            apps[parts[0]] = int(parts[1])
    return apps


def nvidia_used_mb() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def ensure_comfyui_running(label: str) -> bool:
    """Arranca ComfyUI con ESTE proceso si no está vivo (deja el Popen tracked)."""
    from ui.tabs import comfy_launcher
    if port_open(COMFY_PORT):
        print(f"[PROBE] ComfyUI ya activo en {COMFY_PORT} ({label})", flush=True)
        return True
    print(f"[PROBE] ComfyUI no activo en {COMFY_PORT}; arrancando ({label})…", flush=True)
    ok, message, _ = comfy_launcher.start(port=COMFY_PORT, directly_run=True)
    check(f"comfyui_started_{label}", ok, str(message))
    for _ in range(150):
        if port_open(COMFY_PORT):
            break
        time.sleep(1)
    return port_open(COMFY_PORT)


def test_config_is_valid() -> None:
    settings = ss.load_settings()
    for key in ("stop_comfyui_on_close", "release_local_gpu_on_close",
                "abort_local_batch_on_close"):
        check(f"config_{key}_bool", isinstance(settings.get(key), bool),
              f"{key}={settings.get(key)!r}")
    grace = settings.get("queue_grace_seconds")
    check("config_grace_non_negative",
          isinstance(grace, (int, float)) and not isinstance(grace, bool) and grace >= 0,
          str(grace))
    poll = settings.get("poll_seconds")
    check("config_poll_positive",
          isinstance(poll, (int, float)) and not isinstance(poll, bool) and poll > 0,
          str(poll))
    check("session_log_dir_is_project_logs",
          Path(ss._SESSION_LOG_DIR) == ROOT / "logs", ss._SESSION_LOG_DIR)


def test_shutdown_is_idempotent() -> None:
    """La ventana y atexit llaman a la misma función: solo puede ejecutar una vez."""
    from ui.tabs import comfy_launcher
    calls = {"stop": 0, "release": 0}
    saved_stop, saved_release = comfy_launcher.stop, ss._release_local_gpu
    saved_dir = ss._SESSION_LOG_DIR

    def fake_stop():
        calls["stop"] += 1
        return True, "detenido por el probe"

    def fake_release():
        calls["release"] += 1
        return {"attempted": True, "steps": ["probe"], "errors": []}

    comfy_launcher.stop = fake_stop
    ss._release_local_gpu = fake_release
    ss._SESSION_LOG_DIR = str(OUT)
    try:
        reset()
        first = ss.shutdown_session("probe-idempotente")
        second = ss.shutdown_session("probe-idempotente-repetido")
        check("shutdown_runs_actions_once",
              calls["stop"] == 1 and calls["release"] == 1,
              f"stop={calls['stop']} release={calls['release']}")
        check("second_call_returns_cached_report",
              second.get("reason") == first.get("reason"),
              f"reason={second.get('reason')!r}")
        check("shutdown_report_ok", first.get("ok") is True, str(first.get("error")))
        report_path = Path(first.get("report_path") or "")
        check("report_json_written", report_path.is_file(), str(report_path))
        if report_path.is_file():
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            needed = {"before", "after", "comfyui", "local_gpu", "settings"}
            check("report_json_has_state", needed <= set(payload),
                  ",".join(sorted(payload)))
    finally:
        comfy_launcher.stop = saved_stop
        ss._release_local_gpu = saved_release
        ss._SESSION_LOG_DIR = saved_dir
        reset()


def test_invalid_config_refuses_actions() -> None:
    """Config ilegible: avisar en voz alta y NO adivinar política."""
    from ui.tabs import comfy_launcher
    calls = {"stop": 0}
    saved_stop = comfy_launcher.stop
    saved_path, saved_dir = ss._SETTINGS_PATH, ss._SESSION_LOG_DIR
    broken = OUT / "broken_session_shutdown.json"

    def fake_stop():
        calls["stop"] += 1
        return True, "no debería llamarse"

    comfy_launcher.stop = fake_stop
    ss._SESSION_LOG_DIR = str(OUT)
    try:
        broken.write_text("{ no es json", encoding="utf-8")
        ss._SETTINGS_PATH = str(broken)
        reset()
        report = ss.shutdown_session("probe-config-mala")
        check("invalid_config_is_refused",
              report.get("ok") is False and "error" in report,
              str(report.get("error"))[:90])
        check("invalid_config_runs_no_action", calls["stop"] == 0,
              f"stop llamado {calls['stop']} veces")
        check("invalid_config_explains_itself",
              "no se ejecuta ninguna acción" in str(report.get("note", "")),
              str(report.get("note"))[:80])
    finally:
        comfy_launcher.stop = saved_stop
        ss._SETTINGS_PATH = saved_path
        ss._SESSION_LOG_DIR = saved_dir
        reset()


def test_state_observes_real_machine() -> None:
    ensure_comfyui_running("estado")
    state = ss.observe_state()
    needed = {"vram_used_mb", "vram_total_mb", "comfyui_running",
              "comfyui_queue_running", "comfyui_queue_pending", "local_batch_busy"}
    check("state_has_all_fields", needed <= set(state), ",".join(sorted(state)))
    check("state_reads_vram_from_nvidia_smi",
          isinstance(state["vram_used_mb"], int) and state["vram_used_mb"] > 0,
          f"{state['vram_used_mb']}MB usados")
    check("state_sees_live_comfyui", state["comfyui_running"] is True,
          "ComfyUI debería estar vivo antes del test real")
    check("state_sees_idle_queue",
          state["comfyui_queue_running"] == 0 and state["comfyui_queue_pending"] == 0,
          f"cola {state['comfyui_queue_running']}/{state['comfyui_queue_pending']}")
    check("state_sees_no_local_batch", state["local_batch_busy"] is False,
          str(state["local_batch_busy"]))
    check("idle_state_is_not_busy", ss._is_busy(state) is False)


def test_core_is_wired() -> None:
    source = (ROOT / "roop" / "core.py").read_text(encoding="utf-8")
    check("core_registers_atexit_hook", "register_shutdown_hook()" in source)
    check("core_calls_shutdown_on_close",
          'shutdown_session("ventana-cerrada")' in source)
    check("core_imports_shutdown_module",
          "from roop.session_shutdown import register_shutdown_hook, shutdown_session" in source)
    check("core_no_longer_promises_to_keep_comfyui",
          "ComfyUI permanece disponible para otras sesiones" not in source)


def test_local_gpu_release_frees_real_vram() -> None:
    """Residente real del proceso principal: cargar FaceAnalysis y soltarla.

    No es una simulación: se crea una sesión ONNX en la GPU y se comprueba con
    ``nvidia-smi`` que la descarga devuelve la memoria.
    """
    import torch
    if not torch.cuda.is_available():
        check("local_release_cuda_available", False, "CUDA no disponible")
        return
    check("local_release_cuda_available", True, torch.cuda.get_device_name(0))

    from roop import face_util
    analyser = face_util.get_face_analyser()
    check("local_face_analyser_loaded", analyser is not None,
          str(face_util.FACE_ANALYSER_STATUS))
    if analyser is None:
        return

    used_loaded = nvidia_used_mb()
    result = ss._release_local_gpu()
    time.sleep(1.5)
    used_released = nvidia_used_mb()

    check("local_release_has_no_errors", not result.get("errors"),
          str(result.get("errors"))[:90])
    check("local_release_drops_analyser", face_util.FACE_ANALYSER is None,
          str(type(face_util.FACE_ANALYSER)))
    check("local_release_frees_vram",
          isinstance(used_loaded, int) and isinstance(used_released, int)
          and used_released < used_loaded,
          f"{used_loaded}→{used_released}MB (Δ {used_released - used_loaded}MB)")
    check("local_release_reports_steps", bool(result.get("steps")),
          ",".join(result.get("steps") or []))


def _real_scenario(label: str, force_unowned: bool) -> None:
    """End-to-end real: matar ComfyUI y liberar VRAM en el escenario ``label``.

    ``force_unowned=False`` deja el Popen tracked (ventana que arrancó ComfyUI);
    ``force_unowned=True`` lo limpia para reproducir una ventana secundaria que
    reutiliza una sesión ajena. Los dos tienen que acabar igual: puerto cerrado,
    PIDs de GPU fuera y VRAM liberada.
    """
    from ui.tabs import comfy_launcher

    def tag(name: str, condition: bool, detail: str = "") -> None:
        check(f"{name}_{label}", condition, detail)

    # No interrumpir una UI activa ni una cola que está renderizando.
    active_gradio = [port for port in GRADIO_PORTS if port_open(port)]
    tag("no_active_ui", not active_gradio, f"puertos de UI abiertos: {active_gradio}")

    if not force_unowned and comfy_launcher.comfy_process is None and port_open(COMFY_PORT):
        print("[PROBE] ComfyUI vivo pero ajeno: se reinicia con este proceso "
              "para poder probar el escenario propietario", flush=True)
        comfy_launcher.stop()
        time.sleep(2)
    if not ensure_comfyui_running(label):
        tag("comfyui_available", False, f"no arrancó en {COMFY_PORT}")
        return
    if force_unowned:
        comfy_launcher.comfy_process = None  # ventana secundaria / sesión reutilizada
    tag("ownership_matches_scenario",
        (comfy_launcher.comfy_process is None) == bool(force_unowned),
        f"force_unowned={force_unowned} tracked={comfy_launcher.comfy_process}")

    pids_before = listening_pids(COMFY_PORT)
    apps_before = gpu_apps()
    used_before = nvidia_used_mb()
    comfy_vram = sum(apps_before.get(pid, 0) for pid in pids_before)
    tag("comfyui_listening_before", port_open(COMFY_PORT),
        f"pids={pids_before} vram={comfy_vram}MB usados={used_before}MB")
    tag("queue_idle_before", not ss._is_busy(ss.observe_state()))

    saved_dir = ss._SESSION_LOG_DIR
    ss._SESSION_LOG_DIR = str(OUT)
    try:
        reset()
        report = ss.shutdown_session(f"probe-real-{label}")
    finally:
        ss._SESSION_LOG_DIR = saved_dir
        state_after = ss.observe_state()

    comfy_report = report.get("comfyui") or {}
    tag("shutdown_ok", report.get("ok") is True, str(report.get("error")))
    tag("warm_imports_clean",
        not (report.get("warm_imports") or {}).get("failed"),
        str((report.get("warm_imports") or {}).get("failed"))[:90])
    tag("stopped_comfyui", comfy_report.get("stopped") is True,
        str(comfy_report.get("message"))[:90])
    tag("checked_state_first", (report.get("grace") or {}).get("idle") is True,
        f"espera {(report.get('grace') or {}).get('waited_seconds')}s")
    tag("port_closed_after", not port_open(COMFY_PORT),
        f"pids restantes={listening_pids(COMFY_PORT)}")
    tag("pids_gone_from_gpu", not (set(pids_before) & set(gpu_apps().keys())),
        f"antes={pids_before} ahora={sorted(gpu_apps())}")
    tag("local_gpu_release_no_errors",
        not (report.get("local_gpu") or {}).get("errors"),
        str((report.get("local_gpu") or {}).get("errors"))[:90])
    delta = report.get("vram_delta_mb")
    used_after = nvidia_used_mb()
    tag("vram_freed", isinstance(delta, int) and delta < 0,
        f"Δ {delta}MB ({used_before}→{used_after}MB)")
    tag("comfyui_dead_after", state_after["comfyui_running"] is False,
        str(state_after["comfyui_running"]))
    path = Path(report.get("report_path") or "")
    tag("report_json_written", path.is_file(), str(path))
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        tag("report_reason_matches", payload.get("reason") == f"probe-real-{label}",
            str(payload.get("reason")))


def test_atexit_path_kills_comfyui() -> None:
    """La salida por Ctrl+C/excepción también cierra ComfyUI (ruta atexit).

    Se registra el hook en un proceso hijo real y se le deja salir: si el hook
    no se ejecutara, ComfyUI seguiría vivo. Además el informe debe caer en la
    carpeta de producción ``logs/``.
    """
    ensure_comfyui_running("atexit")
    if not port_open(COMFY_PORT):
        check("atexit_comfyui_available", False, "no hay ComfyUI que cerrar")
        return
    check("atexit_comfyui_available", True, f"pids={listening_pids(COMFY_PORT)}")

    reports_before = set((ROOT / "logs").glob("session_shutdown_*.json"))
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from roop.session_shutdown import register_shutdown_hook\n"
        "register_shutdown_hook()\n"
        "print('CHILD REGISTERED', flush=True)\n"
    )
    child = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=300)
    output = child.stdout + child.stderr
    check("atexit_child_registered", "CHILD REGISTERED" in child.stdout,
          child.stdout.strip()[-120:])
    check("atexit_child_exited_clean", child.returncode == 0,
          f"rc={child.returncode}")
    check("atexit_child_ran_shutdown",
          "[SessionShutdown]" in output, output.strip()[-160:].replace("\n", " | "))
    check("atexit_killed_comfyui", not port_open(COMFY_PORT),
          f"pids restantes={listening_pids(COMFY_PORT)}")

    new_reports = set((ROOT / "logs").glob("session_shutdown_*.json")) - reports_before
    newest = sorted(new_reports)[-1] if new_reports else None
    check("atexit_report_written_to_logs", newest is not None and newest.is_file(),
          str(newest))
    if newest:
        payload = json.loads(newest.read_text(encoding="utf-8"))
        check("atexit_report_reason", payload.get("reason") == "atexit",
              str(payload.get("reason")))
        check("atexit_report_ok", payload.get("ok") is True,
              str(payload.get("error")))
        check("atexit_warm_imports_clean",
              not (payload.get("warm_imports") or {}).get("failed"),
              str((payload.get("warm_imports") or {}).get("failed"))[:120])
        check("atexit_report_has_vram",
              isinstance(payload.get("before", {}).get("vram_used_mb"), int),
              str(payload.get("before", {}).get("vram_used_mb")))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    test_config_is_valid()
    test_shutdown_is_idempotent()
    test_invalid_config_refuses_actions()
    test_state_observes_real_machine()
    test_core_is_wired()
    test_local_gpu_release_frees_real_vram()
    # Dos escenarios reales: la ventana que arrancó ComfyUI (propietario) y la
    # ventana secundaria que reutiliza una sesión ajena. Los dos deben matarlo.
    _real_scenario("propietario", force_unowned=False)
    _real_scenario("secundario", force_unowned=True)
    # Tercera ruta real: la salida del proceso (Ctrl+C / excepción) vía atexit.
    test_atexit_path_kills_comfyui()

    (OUT / "results.json").write_text(
        json.dumps({"failures": _failures, "total_failures": len(_failures)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verdict = "PASS" if not _failures else "FAIL"
    print(f"\nSession shutdown: {verdict} — {len(_failures)} fallo(s); "
          f"failures={_failures}", flush=True)
    print(f"EVIDENCE {OUT}", flush=True)
    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())


