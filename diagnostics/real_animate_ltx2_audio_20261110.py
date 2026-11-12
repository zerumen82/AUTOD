"""Prueba real de LTX-2 Q4 con audio nativo en Animate (GPU, 8 GB).

Mide lo que el usuario pidió comprobar de verdad: si el modelo cabe y qué
recursos consume. Registra pico de VRAM (nvidia-smi), pico de RAM del proceso
ComfyUI, duración por fase y demuestra que el MP4 lleva pista de audio.

La ruta está expuesta en Animate (`ltx2_audio`) desde v5.1328; esta prueba es
la medición de recurso que la respalda. El PASS visual con retrato real sigue
pendiente (MEMORY.md, 2026-11-10).

Uso:
    venv/Scripts/python.exe diagnostics/real_animate_ltx2_audio_20261110.py \
        --image "ruta/foto.png" --prompt "..." [--frames 97] [--keep-comfy]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.animate import ltx2_audio as ltx  # noqa: E402

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "ltx2_audio_20261110")
COMFY = os.path.join(ltx.PROJECT_ROOT, "ui", "tob", "ComfyUI")
COMFY_INPUT = os.path.join(COMFY, "input")


def _nvidia_used_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _comfy_pid(port=8188):
    try:
        import psutil
        # Varios procesos comparten cmdline (hijos de nodos custom): el
        # servidor real es el de mayor RSS, no el primero que aparezca.
        candidates = []
        for proc in psutil.process_iter(["pid", "cmdline", "memory_info"]):
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "main.py" in cmd and f"--port {port}" in cmd:
                candidates.append((proc.info["memory_info"].rss, proc.info["pid"]))
        if candidates:
            return max(candidates)[1]
    except Exception:
        pass
    try:
        import psutil
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.status == "LISTEN" and conn.pid:
                return conn.pid
    except Exception:
        pass
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                return int(line.split()[-1])
    except Exception:
        pass
    return None


class ResourceSampler(threading.Thread):
    def __init__(self, pid, interval=1.0):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.peak_gpu = 0
        self.peak_rss = 0
        self.samples = 0
        self._stop_event = threading.Event()

    def run(self):
        import psutil
        proc = psutil.Process(self.pid) if self.pid else None
        while not self._stop_event.is_set():
            used = _nvidia_used_mb()
            if used:
                self.peak_gpu = max(self.peak_gpu, used)
            if proc is not None:
                try:
                    rss = proc.memory_info().rss
                    for child in proc.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except Exception:
                            pass
                    self.peak_rss = max(self.peak_rss, rss)
                except Exception:
                    pass
            self.samples += 1
            time.sleep(self.interval)

    def finish(self):
        # No llamar `stop`: threading.Thread usa `_stop()` internamente y
        # sobrescribirlo rompe join().
        self._stop_event.set()


def _ffprobe_path():
    try:
        from roop.utils import get_ffmpeg_path
        ffmpeg = get_ffmpeg_path()
        candidate = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
        if os.path.exists(candidate):
            return candidate
        if shutil.which("ffprobe"):
            return shutil.which("ffprobe")
    except Exception:
        pass
    return shutil.which("ffprobe")


def probe_media(path):
    info = {"path": path, "bytes": os.path.getsize(path) if os.path.exists(path) else 0}
    ffprobe = _ffprobe_path()
    if not ffprobe:
        info["error"] = "ffprobe no disponible"
        return info
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True, timeout=120,
    )
    try:
        data = json.loads(out.stdout)
    except Exception:
        info["error"] = (out.stderr or out.stdout)[-300:]
        return info
    streams = data.get("streams", [])
    info["duration_s"] = float(data.get("format", {}).get("duration", 0) or 0)
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    info["video"] = [{"codec": s.get("codec_name"), "w": s.get("width"), "h": s.get("height"),
                      "fps": s.get("avg_frame_rate")} for s in video]
    info["audio"] = [{"codec": s.get("codec_name"), "rate": s.get("sample_rate"),
                      "channels": s.get("channels")} for s in audio]
    info["has_audio"] = bool(audio)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--width", type=int, default=ltx.DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=ltx.DEFAULT_HEIGHT)
    ap.add_argument("--frames", type=int, default=ltx.DEFAULT_FRAMES)
    ap.add_argument("--fps", type=int, default=ltx.DEFAULT_FPS)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--keep-comfy", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    log_path = os.path.join(OUT_ROOT, "run.log")
    log = open(log_path, "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    started_comfy = False
    sampler = None
    result = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "prompt": args.prompt,
              "resolution": [args.width, args.height], "frames": args.frames, "fps": args.fps,
              "seed": args.seed}

    missing = ltx.missing_models()
    if missing:
        say(f"[FAIL] Faltan modelos: {', '.join(missing)}")
        result["error"] = "modelos_incompletos"
        json.dump(result, open(os.path.join(OUT_ROOT, "summary.json"), "w"), indent=2)
        return 2

    ltx.ensure_clip_links()
    say("[INFO] Modelos completos y enlaces de CLIP listos")

    try:
        from ui.tabs.comfy_launcher import is_comfyui_running, start as start_comfyui
        if not is_comfyui_running():
            say("[INFO] ComfyUI cerrado; arrancando con los flags del proyecto")
            ok, msg, _port = start_comfyui(port=8188, directly_run=True)
            started_comfy = True
            say(f"[INFO] Arranque ComfyUI: {ok} {msg}")
            if not ok:
                result["error"] = f"comfyui_no_arranca: {msg}"
                json.dump(result, open(os.path.join(OUT_ROOT, "summary.json"), "w"), indent=2)
                return 3
            time.sleep(3)

        import requests
        base = "http://127.0.0.1:8188"
        say("[INFO] /system_stats " + json.dumps(requests.get(base + "/system_stats", timeout=20).json())[:200])
        result["comfy_vram_total_mb"] = _nvidia_used_mb()
        result["vram_baseline_mb"] = _nvidia_used_mb()

        stamp = int(time.time())
        image_name = f"ltx2probe_{stamp}.png"
        from PIL import Image
        img = Image.open(args.image).convert("RGB")
        img.save(os.path.join(COMFY_INPUT, image_name))
        result["input_image"] = args.image
        result["input_size"] = img.size

        workflow = ltx.build_workflow(
            image_name, args.prompt, width=args.width, height=args.height,
            frames=args.frames, fps=args.fps, seed=args.seed,
            filename_prefix=f"animate/ltx2probe_{stamp}",
        )
        json.dump(workflow, open(os.path.join(OUT_ROOT, "workflow.json"), "w"), indent=2)

        pid = _comfy_pid()
        sampler = ResourceSampler(pid)
        sampler.start()
        say(f"[INFO] ComfyUI PID={pid}; enviando prompt ({len(workflow)} nodos)")
        t0 = time.time()
        r = requests.post(base + "/prompt", json={"prompt": workflow, "client_id": uuid.uuid4().hex}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"/prompt {r.status_code}: {r.text[:800]}")
        prompt_id = r.json()["prompt_id"]
        result["prompt_id"] = prompt_id

        history = None
        last_note = 0
        while time.time() - t0 < args.timeout:
            time.sleep(3)
            # v5.1339: un /history lento (ComfyUI ocupado con el VAE decode en
            # el mismo proceso) expiraba y abortaba la prueba entera tras
            # ~15 min de render válido. Se reintenta, igual que hace la ruta de
            # producción en roop/animate/ltx2_audio.py.
            try:
                h = requests.get(f"{base}/history/{prompt_id}", timeout=120).json()
            except requests.exceptions.RequestException:
                continue
            if prompt_id in h:
                history = h[prompt_id]
                break
            if time.time() - last_note > 30:
                try:
                    q = requests.get(base + "/queue", timeout=20).json()
                    running = len(q.get('queue_running', []))
                    pending = len(q.get('queue_pending', []))
                except requests.exceptions.RequestException:
                    running, pending = -1, -1
                say(f"[WAIT] {time.time() - t0:.0f}s cola={running} ejecutando, "
                    f"pendientes={pending}, gpu={_nvidia_used_mb()}MB")
                last_note = time.time()

        elapsed = time.time() - t0
        result["generation_s"] = round(elapsed, 1)
        if history is None:
            raise TimeoutError(f"sin resultado tras {elapsed:.0f}s")

        status = history.get("status", {})
        result["comfy_status"] = status.get("status_str")
        if status.get("status_str") != "success":
            msgs = [m for m in status.get("messages", []) if m[0] in ("execution_error", "execution_interrupted")]
            raise RuntimeError(f"ComfyUI status={status.get('status_str')}: {json.dumps(msgs)[:1200]}")

        outputs = history.get("outputs", {})
        files = []
        for node_out in outputs.values():
            for entry in node_out.get("images", []) + node_out.get("videos", []):
                if entry.get("type") == "output" and entry.get("filename", "").endswith(".mp4"):
                    files.append(entry)
        if not files:
            raise RuntimeError(f"Sin MP4 en outputs: {json.dumps(outputs)[:600]}")

        entry = files[0]
        src = os.path.join(COMFY, "output", entry.get("subfolder", ""), entry["filename"])
        dst = os.path.join(OUT_ROOT, "result.mp4")
        shutil.copy2(src, dst)
        say(f"[OK] MP4 {dst} ({os.path.getsize(dst) / 1e6:.1f} MB) en {elapsed:.0f}s")

        media = probe_media(dst)
        result["media"] = media
        result["has_native_audio"] = bool(media.get("has_audio"))
        say(f"[AUDIO] pista={'sí' if media.get('has_audio') else 'NO'} "
            f"{media.get('audio')} duración={media.get('duration_s')}s vídeo={media.get('video')}")

    except Exception as e:
        result["error"] = str(e)[:2000]
        say(f"[FAIL] {e}")
    finally:
        if sampler is not None:
            sampler.finish()
            sampler.join(timeout=5)
            result["peak_gpu_mb"] = sampler.peak_gpu
            result["peak_comfy_rss_mb"] = round(sampler.peak_rss / 1e6, 1)
            result["samples"] = sampler.samples
            say(f"[VRAM] pico={sampler.peak_gpu} MB / 8192 MB; "
                f"pico RAM ComfyUI={sampler.peak_rss / 1e9:.1f} GB")
        result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        json.dump(result, open(os.path.join(OUT_ROOT, "summary.json"), "w"), indent=2)
        log.close()
        if started_comfy and not args.keep_comfy:
            try:
                from ui.tabs.comfy_launcher import stop as stop_comfyui
                stop_comfyui()
                say("[INFO] ComfyUI cerrado al terminar (devolver el equipo a su estado)")
            except Exception as e:
                say(f"[WARN] No se pudo cerrar ComfyUI: {e}")

    return 0 if not result.get("error") and result.get("has_native_audio") else 1


if __name__ == "__main__":
    sys.exit(main())
