"""Fix FaceSwapTab logs display - wire log_capture into events.py"""
import re

events_path = r'D:\PROJECTS\AUTOAUTO\ui\tabs\faceswap\events.py'
with open(events_path, 'r', encoding='utf-8') as f:
    code = f.read()

if 'log_display' not in code:
    code = code.replace(
        'def on_start_process(fake_preview, auto_rot, temp_smooth, dist, blend, enhancer_blend, enhancer):',
        'def on_start_process(fake_preview, auto_rot, temp_smooth, dist, blend, enhancer_blend, enhancer, log_display):'
    )
    code = code.replace(
        'outputs=[ui_comp["bt_start"], ui_comp["bt_stop"], ui_comp["metrics_display"]]',
        'outputs=[ui_comp["bt_start"], ui_comp["bt_stop"], ui_comp["metrics_display"], ui_comp["log_display"]]'
    )
    old_body = '    import os, sys'
    new_body = """    import os, sys, collections
    # --- Log Capture ---
    _log_lines = []
    def _append_log(msg):
        if msg and msg.strip():
            _log_lines.append(msg.strip())
            if len(_log_lines) > 50:
                del _log_lines[0]
"""
    code = code.replace(old_body, new_body, 1)
    with open(events_path, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f'[OK] {events_path} updated')
else:
    print('[SKIP] already updated')

print('Done!')