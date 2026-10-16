#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe CPU de WsProgress (v5.1205/v5.1206) — sin inferencia ni GPU.

Cubre:
 1. WsProgress.handle_payload con mensajes enlatados del protocolo /ws:
    paso real, fase por nodo, filtros de prompt_id, duraciones y resumen.
 2. seconds_per_step a partir de marcas de tiempo controladas.
 3. Conexión real (solo lectura) al /ws de ComfyUI — sin enviar prompts.
 4. Import del cliente Flux y extracción del mapa nodo→clase del workflow.
 5. v5.1206 — client_id: new_client_id() único, ws_url con clientId
    (requerido: ComfyUI solo emite executing si el POST /prompt lleva
    client_id), "executed" cierra la duración del nodo, firma del
    wait_for_comfy_image con client_id y contrato en los POST de
    flux_edit/flux_gen.

Uso:
    PYTHONPATH=. venv\\Scripts\\python.exe diagnostics/probe_comfy_ws_progress.py
"""
import inspect
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.img_editor.comfy_progress import (  # noqa: E402
    WsProgress,
    _websocket_client,
    new_client_id,
    wait_for_comfy_image,
)

PID = "11111111-2222-3333-4444-555555555555"
NODES = {
    "2": "UnetLoaderGGUF",
    "3": "CLIPLoaderGGUF",
    "4": "VAELoader",
    "6": "TextEncodeQwenImageEditPlus",
    "8": "KSampler",
    "9": "VAEDecode",
    "10": "SaveImage",
}
PASS = 0
FAIL = 0


def check(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label} {detail}")


def canned(mtype, data):
    return {"type": mtype, "data": data}


def test_payload_parser():
    print("[1] Parser de mensajes /ws (enlatados)")
    ws = WsProgress("http://127.0.0.1:8188", PID, NODES)

    ws.handle_payload(canned("execution_start", {"prompt_id": PID}))
    ws.handle_payload(canned("executing", {"prompt_id": PID, "node": "2"}))
    ws.handle_payload(canned("executing", {"prompt_id": PID, "node": "3"}))
    check("carga de modelos como fase loader", ws.phase_label() == "Cargando modelos",
          f"-> {ws.phase_label()!r}")

    ws.handle_payload(canned("executing", {"prompt_id": PID, "node": "6"}))
    check("TextEncode nombra la fase VL",
          ws.phase_label() == "Codificando texto+imagen (Qwen VL)",
          f"-> {ws.phase_label()!r}")

    ws.handle_payload(canned("executing", {"prompt_id": PID, "node": "8"}))
    for i in range(1, 4):
        ws.handle_payload(canned("progress", {"prompt_id": PID, "value": i, "max": 12, "node": "8"}))
    check("paso real del sampler", ws.step == 3 and ws.total_steps == 12,
          f"-> step={ws.step} total={ws.total_steps}")
    check("fase Muestreando paso 3/12", ws.phase_label() == "Muestreando paso 3/12",
          f"-> {ws.phase_label()!r}")

    # Mensaje de OTRO prompt debe ignorarse.
    ws.handle_payload(canned("progress", {"prompt_id": "otro", "value": 9, "max": 12}))
    ws.handle_payload(canned("executing", {"prompt_id": "otro", "node": "99"}))
    check("filtra mensajes de otro prompt_id",
          ws.step == 3 and ws.current_node_id == "8",
          f"-> step={ws.step} node={ws.current_node_id}")

    ws.handle_payload(canned("executing", {"prompt_id": PID, "node": "9"}))
    ws.handle_payload(canned("executing", {"prompt_id": PID, "node": "10"}))
    ws.handle_payload(canned("execution_success", {"prompt_id": PID}))
    check("done al recibir execution_success", ws.done is True)
    check("nodos registrados con nombre de clase",
          ws.node_ms["6"]["name"] == "TextEncodeQwenImageEditPlus"
          and ws.node_ms["8"]["name"] == "KSampler",
          f"-> {ws.node_ms}")

    # Duraciones: en una instancia fresca, retroceder el arranque del nodo
    # simula tiempo real dentro del nodo (los mensajes enlatados llegan con
    # ~0ms de separación).
    ws5 = WsProgress("http://127.0.0.1:8188", PID, NODES)
    ws5.handle_payload(canned("executing", {"prompt_id": PID, "node": "6"}))
    ws5._node_start -= 512.0
    ws5.handle_payload(canned("executing", {"prompt_id": PID, "node": "8"}))
    ws5._node_start -= 96.0
    ws5.handle_payload(canned("executing", {"prompt_id": PID, "node": "9"}))
    ws5.handle_payload(canned("progress", {"prompt_id": PID, "value": 2, "max": 12}))
    ws5._step_ts = [1000.0, 1008.0]  # marcas controladas: 8.0 s/paso
    summary = ws5.summary()
    check("resumen contiene nodos clave con segundos",
          "TextEncodeQwenImageEditPlus 512s" in summary and "KSampler 96s" in summary
          and "8.0s/paso" in summary,
          f"-> {summary!r}")

    print("[2] execution_error marca error")
    ws2 = WsProgress("http://127.0.0.1:8188", PID, NODES)
    ws2.handle_payload(canned("execution_error", {
        "prompt_id": PID, "exception_message": "CUDA out of memory",
    }))
    check("error registrado", ws2.done and "CUDA" in ws2.error)
    check("fase de error", ws2.phase_label() == "Error en ComfyUI")

    print("[3] seconds_per_step con marcas controladas")
    ws3 = WsProgress("http://127.0.0.1:8188", PID, NODES)
    ws3._step_ts = [1000.0, 1008.0, 1016.0, 1024.0]  # 8s por paso
    check("s/paso calculado", abs(ws3.seconds_per_step() - 8.0) < 1e-6,
          f"-> {ws3.seconds_per_step()}")
    ws4 = WsProgress("http://127.0.0.1:8188", PID, NODES)
    check("s/paso sin datos es 0", ws4.seconds_per_step() == 0.0)


def test_client_id_contract():
    print("[4] v5.1206 — client_id (POST /prompt + /ws?clientId=)")
    cid = new_client_id()
    check("new_client_id con prefijo autoauto", cid.startswith("autoauto-"), f"-> {cid}")
    check("new_client_id único entre llamadas", cid != new_client_id())

    # El clientId del /ws debe ser el MISMO valor del POST; sin él,
    # execution.py:483 no emite "executing" y no hay fases ni "Fases:".
    ws = WsProgress("http://127.0.0.1:8188", PID, NODES, client_id=cid)
    check("ws_url usa el client_id del POST",
          ws.ws_url == f"ws://127.0.0.1:8188/ws?clientId={cid}",
          f"-> {ws.ws_url}")
    ws_def = WsProgress("http://127.0.0.1:8188", PID, NODES)
    check("client_id por defecto derivado del prompt_id",
          ws_def.client_id == f"autoauto-{PID[:12]}",
          f"-> {ws_def.client_id}")

    # "executed" cierra YA la duración del nodo actual.
    ws2 = WsProgress("http://127.0.0.1:8188", PID, NODES, client_id=cid)
    ws2.handle_payload(canned("executing", {"prompt_id": PID, "node": "6"}))
    ws2._node_start -= 42.0
    ws2.handle_payload(canned("executed", {"prompt_id": PID, "node": "6"}))
    check("executed cierra el nodo actual",
          ws2.node_ms["6"]["ms"] >= 41.9 and ws2._node_start is None,
          f"-> ms={ws2.node_ms['6']['ms']} start={ws2._node_start}")
    # executed de OTRO nodo no debe tocar el nodo en curso.
    ws2.handle_payload(canned("executing", {"prompt_id": PID, "node": "8"}))
    ws2.handle_payload(canned("executed", {"prompt_id": PID, "node": "6"}))
    check("executed ajeno no cierra el nodo en curso",
          ws2.current_node_id == "8" and ws2._node_start is not None,
          f"-> node={ws2.current_node_id} start={ws2._node_start}")

    # Firma: wait_for_comfy_image debe aceptar client_id.
    sig = inspect.signature(wait_for_comfy_image)
    check("wait_for_comfy_image acepta client_id", "client_id" in sig.parameters)

    # Contrato en los clientes: POST con client_id y pase al waiter.
    import roop.img_editor.flux_edit_comfy_client as edit_mod  # noqa: E402
    import roop.img_editor.flux_gen_comfy_client as gen_mod  # noqa: E402
    for mod, name in ((edit_mod, "flux_edit"), (gen_mod, "flux_gen")):
        src = inspect.getsource(mod)
        check(f"{name}: POST /prompt con client_id",
              '"client_id": client_id' in src, "-> falta en el POST")
        check(f"{name}: wait_for_comfy_image recibe client_id",
              "client_id=client_id" in src, "-> no se pasa al waiter")


def test_live_ws_connect():
    print("[5] Conexión real al /ws de ComfyUI (solo lectura)")
    if _websocket_client is None:
        print("  SKIP  websocket-client no disponible")
        return
    import requests
    try:
        requests.get("http://127.0.0.1:8188/system_stats", timeout=5)
    except Exception as exc:
        print(f"  SKIP  ComfyUI no responde: {exc}")
        return
    ws = WsProgress("http://127.0.0.1:8188", "pid-que-no-existe", NODES)
    ws.start()
    deadline = time.time() + 8
    while time.time() < deadline and not ws.connected:
        time.sleep(0.2)
    check("socket conectado al /ws", ws.connected is True)
    ws.handle_payload(canned("status", {"status": {"exec_info": {"queue_remaining": 0}}}))
    check("status no rompe", ws.done is False)
    ws.stop()
    check("stop() cierra sin excepción", True)


def test_flux_client_node_map():
    print("[6] Import del cliente Flux + mapa nodo→clase")
    from roop.img_editor.flux_edit_comfy_client import FluxEditComfyClient  # noqa: E402
    check("import del cliente OK", FluxEditComfyClient is not None)
    wf = {
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {}},
        "8": {"class_type": "KSampler", "inputs": {}},
    }
    node_names = {str(k): (v or {}).get("class_type", "") for k, v in wf.items()}
    check("extracción id→class_type", node_names == {"2": "UnetLoaderGGUF", "8": "KSampler"},
          f"-> {node_names}")


if __name__ == "__main__":
    test_payload_parser()
    test_client_id_contract()
    test_live_ws_connect()
    test_flux_client_node_map()
    print(f"\nRESULTADO: {PASS} PASS / {FAIL} FAIL")
    sys.exit(1 if FAIL else 0)
