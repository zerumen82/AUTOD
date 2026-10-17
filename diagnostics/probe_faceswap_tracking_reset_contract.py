"""Execute production tracking methods with declared state fixtures, no inference.

AST extraction preserves the exact production bodies and avoids initializing
GPU/model modules. These checks prove cache transitions, not visual tracking.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import traceback
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np

from roop.faceswap_config import get_faceswap_config


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    '_reset_tracking_on_scene_cut', '_clear_video_temporal_mouth_and_occlusion',
    '_mark_unassigned', '_open_reacquire_window', '_should_hold_previous_swap',
    '_has_swap_history', '_video_source_lock_keys',
    'process_frame', '_detect_visual_scene_cut', '_faded_hold_blend',
    '_hold_previous_face_region', '_as_ref_list', '_photo_refs_for_exact_file',
}
ORAL = (
    '_mouth_blend_ema_by_person', '_mouth_force_paste_by_person',
    '_mouth_paste_ema_by_person', '_mouth_state_history', '_mouth_ratio_smooth',
    '_mouth_hold_count',
)


def manager_class(path):
    raw = path.read_bytes()
    module = ast.parse(raw.decode('utf-8-sig'), filename=str(path))
    source_class = next(n for n in module.body if isinstance(n, ast.ClassDef)
                        and n.name == 'ProcessMgr')
    source_class.body = [n for n in source_class.body
                         if isinstance(n, ast.FunctionDef)
                         and n.name in METHODS | {'_destination_match_threshold'}]
    assert METHODS <= {n.name for n in source_class.body}
    source_class.decorator_list = []
    isolated = ast.Module(body=[source_class], type_ignores=[])
    namespace = {'os': os, 'cv2': cv2, 'np': np, 'traceback': traceback,
                 'get_faceswap_config': get_faceswap_config,
                 'roop': SimpleNamespace(globals=SimpleNamespace(selected_face_references={}))}
    quality = ast.parse((ROOT / 'roop' / 'quality_enhancements.py').read_text(encoding='utf-8'))
    histogram = next(n for n in quality.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'match_color_histogram')
    exec(compile(ast.Module(body=[histogram], type_ignores=[]), str(ROOT / 'roop' / 'quality_enhancements.py'), 'exec'), namespace)
    log = ROOT / 'diagnostics' / 'out' / 'tracking_reset_fix_20261006' / 'tester_process_frame.log'

    def diagnostic_open(path, *args, **kwargs):
        # Redirect only process_frame's debug logger into the diagnostic folder.
        return builtins.open(log if str(path).endswith('debug_swap.log') else path, *args, **kwargs)

    namespace['open'] = diagnostic_open
    exec(compile(isolated, str(path), 'exec'), namespace)
    return namespace['ProcessMgr'], hashlib.sha256(raw).hexdigest()


def seed(cls, closeup):
    manager = cls()
    for field in ORAL:
        setattr(manager, field, {
            'video_a.mp4_p0': object(), 'video_a.mp4_p2': object(),
            'video_b.mp4_p0': object(), 'video_a.mp4.part_p0': object(),
        })
    for field in ('_occ_frame_ema', '_prev_frame_result', '_prev_face_bbox',
                  '_prev_face_kps'):
        setattr(manager, field, {'video_a.mp4': object(), 'video_b.mp4': object()})
    for field in ('_last_swap_bbox', '_last_swap_mask'):
        setattr(manager, field, {'a.mp4': object(), 'b.mp4': object()})
    for name in ('a.mp4', 'b.mp4'):
        for field, value in (
            ('_target_face_assigned_', object()), ('_tracking_lost_count_', 5),
            ('_position_history_', [object()]), ('_consecutive_tracking_success_', 4),
            ('_strict_recovery_tick_', 2), ('_last_known_bbox_', object()),
            ('_last_known_kps_', object()), ('_frame_count_', 100),
            ('_reference_face_selected_', object()), ('_tracking_embedding_', object()),
        ):
            setattr(manager, field + name, value)
        setattr(manager, '_profile_hysteresis_video_' + name, object())
        for lock in manager._video_source_lock_keys(name):
            setattr(manager, lock, object())
    manager.selected_face_references = {'a.mp4': object(), 'b.mp4': object()}
    manager.source_face = object()
    manager._mouth_blend_mask = object()
    manager._mouth_blend_factor = .65
    setattr(manager, '_closeup_active_a.mp4', closeup)
    setattr(manager, '_closeup_active_b.mp4', True)
    setattr(manager, '_scene_state_a.mp4', {'last_scene_cut': 20, 'total_scene_changes': 2})
    return manager


def run_variant(cls):
    results = []

    def check(label, condition):
        results.append({'check': label, 'pass': bool(condition)})

    for closeup in (False, True):
        for soft in (False, True, None):
            manager = seed(cls, closeup)
            label = f'closeup={closeup}/soft={soft if soft is not None else "default"}'
            before = dict(manager.__dict__)
            map_before = {k: dict(v) for k, v in before.items() if isinstance(v, dict)}
            kwargs = {} if soft is None else {'soft': soft}
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                manager._reset_tracking_on_scene_cut('a.mp4', 100, reason='contract fixture', **kwargs)
            hard = soft is not True
            check(label + '/reset assignment', getattr(manager, '_target_face_assigned_a.mp4') is None)
            for field in ('_tracking_lost_count_', '_consecutive_tracking_success_', '_strict_recovery_tick_'):
                check(label + '/' + field, getattr(manager, field + 'a.mp4') == 0)
            check(label + '/position reset', getattr(manager, '_position_history_a.mp4') == [])
            for field in ORAL:
                value = getattr(manager, field)
                for key in ('video_a.mp4_p0', 'video_a.mp4_p2'):
                    check(label + '/' + field + '/' + key,
                          key not in value if hard else value.get(key) is map_before[field][key])
                for key in ('video_b.mp4_p0', 'video_a.mp4.part_p0'):
                    check(label + '/isolation/' + field + '/' + key,
                          value.get(key) is map_before[field][key])
            for field, key in (
                ('_occ_frame_ema', 'video_a.mp4'), ('_prev_frame_result', 'video_a.mp4'),
                ('_last_swap_bbox', 'a.mp4'), ('_last_swap_mask', 'a.mp4'),
            ):
                value = getattr(manager, field)
                check(label + '/' + field, key not in value if hard else value.get(key) is map_before[field][key])
            for field in ('_prev_face_bbox', '_prev_face_kps'):
                check(label + '/' + field, 'video_a.mp4' not in getattr(manager, field))
            for field in ('_last_known_bbox_a.mp4', '_last_known_kps_a.mp4'):
                check(label + '/' + field, not hasattr(manager, field))
            profile = '_profile_hysteresis_video_a.mp4'
            check(label + '/profile', not hasattr(manager, profile) if hard else getattr(manager, profile) is before[profile])
            for field, value in before.items():
                if field.endswith('b.mp4') or field.startswith('_video_locked_'):
                    check(label + '/untouched/' + field, getattr(manager, field) is value)
            for field, values in map_before.items():
                if field == '_scene_state_a.mp4':
                    continue
                for key, value in values.items():
                    if key in ('b.mp4', 'video_b.mp4'):
                        check(label + '/untouched/' + field + '/' + key, getattr(manager, field).get(key) is value)
            for field in ('_reference_face_selected_a.mp4', '_tracking_embedding_a.mp4',
                          'selected_face_references', 'source_face', '_mouth_blend_mask'):
                check(label + '/identity_or_instance/' + field, getattr(manager, field) is before[field])
            check(label + '/scene bookkeeping', getattr(manager, '_scene_state_a.mp4') == {'last_scene_cut': 100, 'total_scene_changes': 3})
            check(label + '/suppression', getattr(manager, '_hold_suppressed_until_a.mp4') == 100 + (6 if hard else 2))
            check(label + '/hard marker', getattr(manager, '_hard_scene_cut_frame_a.mp4', None) == (100 if hard else None))
            check(label + '/reacquire', getattr(manager, '_reacquire_window_until_a.mp4', None) == (160 if hard else None))
            check(label + '/plan smooth', getattr(manager, '_plan_change_smooth_until_a.mp4', None) == (124 if soft is True and closeup else None))
            check(label + '/diagnostic tag', ('reset tracking, sin hold' if hard else 'soft, mantiene swap') in captured.getvalue())
            if hard:
                for frame in (100, 101, 114):
                    setattr(manager, '_frame_count_a.mp4', frame)
                    check(label + f'/no hold frame={frame}', manager._should_hold_previous_swap('a.mp4') is False)
                check(label + '/no cached previous swap', manager._has_swap_history('a.mp4') is False)
            elif closeup:
                check(label + '/zoom retains available hold', manager._should_hold_previous_swap('a.mp4') is True)

    cold = cls()
    cold._reset_tracking_on_scene_cut('', 100, soft=False)
    check('empty video ignored', cold.__dict__ == {})
    cold._reset_tracking_on_scene_cut('a.mp4', 100, soft=False)
    check('cold hard reset succeeds', getattr(cold, '_reacquire_window_until_a.mp4') == 160)
    sequence = seed(cls, True)
    sequence._reset_tracking_on_scene_cut('a.mp4', 90, soft=True)
    sequence._reset_tracking_on_scene_cut('a.mp4', 100, soft=False)
    check('zoom then hard cut drops cached frame', 'video_a.mp4' not in sequence._prev_frame_result)
    check('zoom then hard cut drops oral history', 'video_a.mp4_p0' not in sequence._mouth_paste_ema_by_person)
    return results


@contextlib.contextmanager
def empty_detector_fixture():
    """Force the no-detection precondition; never describe this as inference."""
    calls = []
    rotation = ModuleType('roop.face_util_rotation')

    def no_faces(frame, **kwargs):
        calls.append({'shape': list(frame.shape), 'arguments': kwargs})
        return []

    rotation.get_all_faces_smart = no_faces
    rotation.get_face_analyser = lambda: None
    rotation.convert_faces_to_face_objects = lambda faces: faces
    util = ModuleType('roop.face_util')
    util.detect_faces_robust = lambda *args: []
    util.preprocess_image_for_detection = lambda frame: frame
    replacements = {'roop.face_util_rotation': rotation, 'roop.face_util': util}
    originals = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        yield calls
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def frame_fixture(cls, closeup, cut, photo=False):
    manager = seed(cls, closeup)
    manager.is_initialized = True
    manager.processors = {'faceswap': object()}
    manager.input_facesets = [object()]
    manager.options = SimpleNamespace(current_video_path=None if photo else 'a.mp4')
    yy, xx = np.indices((96, 128))
    previous = np.repeat((16 + (xx % 13) + yy % 9).astype(np.uint8)[..., None], 3, axis=2)
    frame = np.full_like(previous, 230) if cut else previous.copy()
    previous_swap = np.repeat((100 + xx % 41 + yy % 31).astype(np.uint8)[..., None], 3, axis=2)
    setattr(manager, '_prev_scene_thumb_a.mp4', cv2.resize(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA))
    setattr(manager, '_scene_diff_history_a.mp4', [0., 1., 0., 1.])
    mask = np.zeros(previous.shape[:2], dtype=np.float32)
    mask[16:80, 24:104] = 1.
    manager._prev_frame_result['video_a.mp4'] = previous_swap
    manager._last_swap_bbox['a.mp4'] = [24, 16, 104, 80]
    manager._last_swap_mask['a.mp4'] = mask
    with empty_detector_fixture() as calls, contextlib.redirect_stdout(io.StringIO()) as logs:
        result = manager.process_frame(frame, enable_temporal_smoothing=True, file_path='control.jpg' if photo else 'a.mp4')
    return {
        'output': result,
        'input': frame,
        'changed_pixels': int(np.count_nonzero(np.any(result != frame, axis=2))),
        'clock': getattr(manager, '_frame_count_a.mp4'),
        'hard_cut': getattr(manager, '_hard_scene_cut_frame_a.mp4', None),
        'previous_frame_cached': 'video_a.mp4' in manager._prev_frame_result,
        'oral_history_cached': 'video_a.mp4_p0' in manager._mouth_paste_ema_by_person,
        'detector_calls': calls,
        'logs': logs.getvalue(),
    }


def frame_contracts(classes):
    evidence = {}
    checks = []
    for closeup in (False, True):
        for cut in (False, True):
            name = f'closeup={closeup}/cut={cut}'
            pair = {label: frame_fixture(cls, closeup, cut) for label, cls in classes.items()}
            before, after = pair['baseline'], pair['current']
            checks.append({'check': name + '/one detector call', 'pass': all(len(v['detector_calls']) == 1 for v in pair.values())})
            checks.append({'check': name + '/clock advances once', 'pass': after['clock'] == 101})
            if cut:
                checks.extend([
                    {'check': name + '/baseline visibly holds prior pixels', 'pass': before['changed_pixels'] > 0},
                    {'check': name + '/current returns original frame', 'pass': after['changed_pixels'] == 0},
                    {'check': name + '/hard cut reported', 'pass': after['hard_cut'] == 101 and 'cambio visual sin cara' in after['logs']},
                    {'check': name + '/drops prior frame and oral history', 'pass': not after['previous_frame_cached'] and not after['oral_history_cached']},
                ])
            else:
                checks.extend([
                    {'check': name + '/continuity pixels identical to baseline', 'pass': np.array_equal(before['output'], after['output'])},
                    {'check': name + '/continues existing hold', 'pass': after['changed_pixels'] > 0},
                    {'check': name + '/history retained', 'pass': after['previous_frame_cached'] and after['oral_history_cached']},
                    {'check': name + '/no hard cut', 'pass': after['hard_cut'] is None},
                ])
            evidence[name] = {label: {k: v for k, v in value.items() if k not in ('output', 'input')} for label, value in pair.items()}
    pair = {label: frame_fixture(cls, True, True, photo=True) for label, cls in classes.items()}
    checks.extend([
        {'check': 'photo/no detections returns original both versions', 'pass': all(v['changed_pixels'] == 0 for v in pair.values())},
        {'check': 'photo/does not advance video clock', 'pass': all(v['clock'] == 100 for v in pair.values())},
        {'check': 'photo/no video reset', 'pass': all(v['hard_cut'] is None and v['oral_history_cached'] for v in pair.values())},
    ])
    evidence['photo'] = {label: {k: v for k, v in value.items() if k not in ('output', 'input')} for label, value in pair.items()}
    return {'scope': 'real process_frame, visual scene detector and hold composition; detector explicitly returns empty as regression fixture; images are generated pixel patterns without faces', 'checks': checks, 'cases': evidence, 'pass': sum(c['pass'] for c in checks), 'fail': sum(not c['pass'] for c in checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = {'scope': 'exact production method bodies executed on declared cache fixtures; no face models, images or video inference', 'variants': {}}
    classes = {}
    for label, path in (('baseline', args.baseline), ('current', ROOT / 'roop' / 'ProcessMgr.py')):
        cls, digest = manager_class(path)
        classes[label] = cls
        checks = run_variant(cls)
        report['variants'][label] = {'path': str(path), 'sha256': digest, 'pass': sum(r['pass'] for r in checks), 'fail': sum(not r['pass'] for r in checks), 'checks': checks}
    before = report['variants']['baseline']
    after = report['variants']['current']
    regression_reproduced = any(not r['pass'] and r['check'].startswith('closeup=True/soft=False/') for r in before['checks'])
    report['baseline_regression_reproduced'] = regression_reproduced
    report['integration'] = frame_contracts(classes)
    report['status'] = 'PASS' if regression_reproduced and after['fail'] == 0 and report['integration']['fail'] == 0 else 'FAIL'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('variants', 'integration')}, ensure_ascii=False))
    print(f"baseline: {before['pass']} PASS / {before['fail']} FAIL; current: {after['pass']} PASS / {after['fail']} FAIL")
    print(f"process_frame: {report['integration']['pass']} PASS / {report['integration']['fail']} FAIL")
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
