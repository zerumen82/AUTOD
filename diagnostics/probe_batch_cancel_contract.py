"""Contratos Fase 0: restauración de stdout/log en core, propiedad de
banderas de cancelación en Generate/Animate y helpers de máscara.

Uso: venv\\Scripts\\python.exe diagnostics/probe_batch_cancel_contract.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import roop.globals  # noqa: E402
from PIL import Image  # noqa: E402

OK = 0
FAIL = 0


def check(name, cond):
    global OK, FAIL
    if cond:
        OK += 1
        print(f"  OK  {name}")
    else:
        FAIL += 1
        print(f" FAIL {name}")


def snapshot_batch_logs():
    log_dir = os.path.join(ROOT, 'logs')
    return {
        f for f in os.listdir(log_dir)
        if f.startswith('process_') and f.endswith('.log')
    } if os.path.isdir(log_dir) else set()


def newest_batch_log(created_before):
    log_dir = os.path.join(ROOT, 'logs')
    new = sorted(
        (f for f in snapshot_batch_logs() - created_before),
        key=lambda f: os.path.getmtime(os.path.join(log_dir, f)),
    )
    if not new:
        return None
    with open(os.path.join(log_dir, new[-1]), encoding='utf-8', errors='replace') as fh:
        return fh.read()


def run_core_case(label, facesets, processing_after_first_yield):
    """Ejecuta batch_process_regular con ProcessMgr mockeado y devuelve
    (outputs, stdout_restaurado, log_con_END)."""
    from roop.core import batch_process_regular

    tmp = tempfile.mkdtemp(prefix='probe_batch_contract_')
    photo = os.path.join(tmp, 'a.png')
    Image.new('RGB', (8, 8), 'red').save(photo)
    photo2 = os.path.join(tmp, 'b.png')
    Image.new('RGB', (8, 8), 'blue').save(photo2)

    entries = [SimpleNamespace(filename=photo), SimpleNamespace(filename=photo2)]
    before = sys.stdout
    created = snapshot_batch_logs()
    outputs = []

    with patch('roop.ProcessMgr.ProcessMgr') as mgr_cls, \
         patch.object(roop.globals, 'output_path', tmp, create=True), \
         patch.object(roop.globals, 'processing', True, create=True), \
         patch.object(roop.globals, 'INPUT_FACESETS', facesets, create=True), \
         patch.object(roop.globals, 'TARGET_FACES', [], create=True), \
         patch.object(roop.globals, 'face_swap_mode', 'selected_faces', create=True):
        mgr_cls.return_value.process_frame.return_value = None
        gen = batch_process_regular(entries, None, '', False, None, 1, None, 0, False)
        try:
            outputs.append(next(gen))          # primer yield (progreso o error)
            if processing_after_first_yield:
                roop.globals.processing = False
            outputs.extend(gen)                # consumir el resto
        except StopIteration:
            pass
        finally:
            gen.close()

    stdout_ok = sys.stdout is before
    log_text = newest_batch_log(created) or ""
    for f in snapshot_batch_logs() - created:
        try:
            os.remove(os.path.join(ROOT, 'logs', f))
        except OSError:
            pass
    print(f"[{label}] outputs={outputs}")
    return outputs, stdout_ok, '=== END' in log_text


# [1] Salida temprana tras instalar el _Tee (sin caras de origen)
outs, restored, has_end = run_core_case('error-sin-fuentes', [], False)
check('core: ruta ERROR devuelve (100, ...)', outs and outs[-1][0] == 100 and 'ERROR' in outs[-1][1])
check('core: stdout restaurado en return temprano', restored)
check('core: log de lote cerrado con END en return temprano', has_end)

# [2] Salida por cancelación dentro del bucle
outs, restored, has_end = run_core_case(
    'cancelacion', [SimpleNamespace(faces=[])], True,
)
check('core: cancelación dentro del bucle', outs and 'cancelado' in outs[-1][1].lower())
check('core: stdout restaurado en cancelación', restored)
check('core: log de lote cerrado con END en cancelación', has_end)

# [3] Generate: el worker es dueño de las banderas
src_gen = open(os.path.join(ROOT, 'ui', 'tabs', 'gen', 'events.py'), encoding='utf-8').read()
worker_seg = src_gen.split('def worker():')[1].split('thread = threading.Thread')[0]
check('gen: worker limpia la cancelación', 'clear_cancel(SCOPE_GENERATION)' in worker_seg)
check('gen: worker limpia is_generating', 'state.is_generating = False' in worker_seg)
check('gen: banderas antes de put(None)', worker_seg.index('state.is_generating = False') < worker_seg.index('updates.put(None)'))
check('gen: finally del generador guardado por worker_started', 'if not worker_started:\n                clear_cancel(SCOPE_GENERATION)' in src_gen)
check('gen: outer finally guardado', 'if not worker_started:\n            state.is_generating = False' in src_gen)
check('gen: guard ANTES de clear_cancel en la entrada',
      src_gen.index('if state.is_generating:') < src_gen.index('clear_cancel(SCOPE_GENERATION)', src_gen.index('def on_generate_image')))

# [4] Animate: el worker es dueño de las banderas
src_ani = open(os.path.join(ROOT, 'ui', 'tabs', 'animate', 'events.py'), encoding='utf-8').read()
worker_ani = src_ani.split('def worker():')[1].split('thread = threading.Thread')[0]
check('animate: worker limpia la cancelación', 'clear_cancel(SCOPE_ANIMATE)' in worker_ani)
check('animate: worker limpia is_animating', 'state.is_animating = False' in worker_ani)
check('animate: banderas antes de put(None)', worker_ani.index('state.is_animating = False') < worker_ani.index('updates.put(None)'))
cancel_seg = src_ani.split('def on_cancel_animate():')[1].split('def wire_animate_events')[0]
check('animate: cancel NO toca is_animating', 'is_animating = False' not in cancel_seg)
check('animate: guard ANTES de clear_cancel en la entrada',
      src_ani.index('if state.is_animating:') < src_ani.index('clear_cancel(SCOPE_ANIMATE)', src_ani.index('def on_animate_click')))
check('animate: el generador ya no limpia cancelación',
      'finally:\n            clear_cancel(SCOPE_ANIMATE)' not in src_ani)

# [5] ProcessMgr: reintento de vídeo blindado
src_pm = open(os.path.join(ROOT, 'roop', 'ProcessMgr.py'), encoding='utf-8').read()
check('pm: defaults de reintento antes del try',
      'fps_video = fps if fps else 24.0' in src_pm and 'total_frames_to_process = 0' in src_pm)
check('pm: guard de prerrequisitos del reintento',
      'if cap is None or fourcc is None or temp_output is None or end_frame is None:' in src_pm)
check('pm: reintento llama a process_frame con file_path',
      'self.process_frame(frame, enable_temporal_smoothing=temporal_smoothing, file_path=video_path)' in src_pm)
check('pm: out.release blindado', 'if out is not None:\n                    out.release()' in src_pm)

# [6] ProcessMgr: gate de segmentación ausente usa hold medido + contador
gate_seg = src_pm.split('if final_mask is None:')[1].split('# D. Aplicar oclusión')[0]
check('pm: gate sin máscara usa hold del frame anterior', '_hold_previous_face_region' in gate_seg)
check('pm: gate sin máscara respeta cortes de escena', '_should_hold_previous_swap' in gate_seg)
check('pm: gate sin máscara cuenta el fallo', '_register_mask_skip' in gate_seg)
check('pm: segundo gate (contorno insuficiente) cuenta el fallo',
      src_pm.count('self._register_mask_skip()') >= 2)
check('pm: contador se reinicia en run_batch_inmem', 'self._mask_skip_count = 0' in src_pm)
check('pm: entrega informa del contador', '_mask_skip_note()' in src_pm)

# [7] Helpers de conteo
from roop.ProcessMgr import ProcessMgr  # noqa: E402
mgr = ProcessMgr.__new__(ProcessMgr)
check('pm: nota vacía sin fallos', mgr._mask_skip_note() == '')
mgr._register_mask_skip()
mgr._register_mask_skip()
check('pm: nota con 2 fallos', mgr._mask_skip_note() == ' | 2 cara(s) sin segmentación (sin swap)')

print()
print(f"RESULTADO: {OK} OK / {FAIL} FAIL")
sys.exit(1 if FAIL else 0)
