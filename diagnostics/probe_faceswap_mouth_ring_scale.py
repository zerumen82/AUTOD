"""Actual helper/gate source contracts using explicit synthetic mask fixtures.

No models, production edits or renders. Config fixtures exercise validation in
an isolated execution namespace; production configuration is never overwritten.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from roop.faceswap_config import get_faceswap_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'diagnostics/out/mouth_math_review_20261006'
CURRENT = ROOT / 'roop/ProcessMgr.py'
BEFORE = OUT / 'ProcessMgr_before.txt'
current_tree = ast.parse(CURRENT.read_text(encoding='utf-8-sig'))
before_tree = ast.parse(BEFORE.read_text(encoding='utf-8-sig'))
helper = deepcopy(next(node for node in ast.walk(current_tree) if isinstance(node, ast.FunctionDef)
                       and node.name == '_mouth_occlusion_ring'))
helper.decorator_list = []
namespace = dict(np=np, cv2=cv2, get_faceswap_config=get_faceswap_config)
exec(compile(ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])), str(CURRENT), 'exec'), namespace)
measure = namespace['_mouth_occlusion_ring']
checks = []
measurements = []


def check(name, condition, detail=None):
    checks.append(dict(name=name, passed=bool(condition), detail=detail))
    print(('OK ' if condition else 'FAIL ') + name + (f': {detail}' if detail is not None else ''))


def assigned_node(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


gate_node = assigned_node(current_tree, '_mouth_has_object')
gate_code = compile(ast.Module(body=[gate_node], type_ignores=[]), 'actual_object_gate', 'exec')
check('actual object acceptance gate unchanged', ast.dump(gate_node) == ast.dump(assigned_node(before_tree, '_mouth_has_object')))
for name in ('_axis_object', 'mouth_paste_strength_raw', 'mouth_low_quality'):
    check(f'{name} unchanged', ast.dump(assigned_node(current_tree, name)) == ast.dump(assigned_node(before_tree, name)))

legacy_statements = [assigned_node(before_tree, name)
                     for name in ('_lips_bin', '_wide', '_inner', '_ring', '_ring_sum')]
legacy_value = next(node for node in ast.walk(before_tree) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'mouth_occ_ring' for target in node.targets)
                    and '_ring_sum' in ast.unparse(node.value))
legacy_code = compile(ast.Module(body=legacy_statements + [legacy_value], type_ignores=[]), str(BEFORE), 'exec')


def old_measure(occ, lips):
    scope = dict(cv2=cv2, np=np, _occ_f=occ, _mm_f=lips)
    exec(legacy_code, scope)
    return scope['mouth_occ_ring']


def gate(occ, lips, ring):
    overlap = occ * lips
    scope = dict(_axis_object=False, mouth_occ_score=float(overlap.mean()),
                 mouth_occ_rel=float(overlap.sum() / lips.sum()), mouth_occ_ring=ring)
    exec(gate_code, scope)
    return scope['_mouth_has_object']


def rejects(name, occ, lips, native):
    try:
        measure(occ, lips, native)
    except (ValueError, TypeError, KeyError, OverflowError):
        check(name, True)
    else:
        check(name, False, 'invalid input accepted')


lips = np.zeros((256, 256), dtype=np.float32)
cv2.ellipse(lips, (128, 170), (40, 10), 0, 0, 360, 1, -1)
occ = cv2.dilate((lips * 255).astype(np.uint8), np.ones((7, 7), np.uint8))
occ = cv2.GaussianBlur(occ, (11, 11), 0).astype(np.float32) / 255
copy_occ, copy_lips = occ.copy(), lips.copy()
native_score = measure(occ, lips, (256, 256))
check('native fixture keeps exact legacy arithmetic', native_score == old_measure(occ, lips),
      dict(current=native_score, legacy=old_measure(occ, lips)))
check('inputs not mutated', np.array_equal(occ, copy_occ) and np.array_equal(lips, copy_lips))
rng = np.random.default_rng(4)
for index in range(12):
    random_occ = rng.random(lips.shape, dtype=np.float32)
    expected = old_measure(random_occ, lips)
    actual = measure(random_occ, lips, lips.shape)
    check(f'native random arithmetic unchanged {index}', actual == expected, dict(current=actual, legacy=expected))

broad = np.zeros_like(lips)
cv2.ellipse(broad, (128, 170), (80, 40), 0, 0, 360, 1, -1)
for width, height in ((256, 256), (512, 512), (512, 256), (256, 512), (768, 512)):
    current_lips = cv2.resize(lips, (width, height), interpolation=cv2.INTER_LINEAR)
    current_occ = cv2.resize(occ, (width, height), interpolation=cv2.INTER_LINEAR)
    score = measure(current_occ, current_lips, lips.shape)
    legacy = old_measure(current_occ, current_lips)
    accepted = bool(gate(current_occ, current_lips, score))
    check(f'lip-only expanded footprint remains rejected {width}x{height}', not accepted)
    check(f'ring scale invariance including anisotropy {width}x{height}', abs(score - native_score) < .015,
          dict(current=score, native=native_score))
    if width == height == 512:
        check('legacy false-to-true scale counterexample reproduced', bool(gate(current_occ, current_lips, legacy)))
    current_broad = cv2.resize(broad, (width, height), interpolation=cv2.INTER_LINEAR)
    broad_score = measure(current_broad, current_lips, lips.shape)
    check(f'broad synthetic footprint remains accepted {width}x{height}', bool(gate(current_broad, current_lips, broad_score)))
    measurements.append(dict(shape=[height, width], current_ring=score, legacy_ring=legacy,
                             current_gate=accepted, legacy_gate=bool(gate(current_occ, current_lips, legacy)),
                             broad_ring=broad_score))

check('zero occlusion gives zero ring', measure(np.zeros_like(lips), lips, lips.shape) == 0)
check('empty lips give zero ring', measure(occ, np.zeros_like(lips), lips.shape) == 0)
check('full lips with no outer ring give zero', measure(occ, np.ones_like(lips), lips.shape) == 0)
rejects('mask shape mismatch rejected', occ, lips[:100], lips.shape)
rejects('empty masks rejected', np.zeros((0, 0)), np.zeros((0, 0)), lips.shape)
rejects('3D masks rejected', occ[..., None], lips[..., None], lips.shape)
rejects('nonfinite occlusion rejected', np.full_like(lips, np.nan), lips, lips.shape)
rejects('nonfinite lips rejected', occ, np.full_like(lips, np.inf), lips.shape)
rejects('negative occlusion rejected', np.full_like(lips, -.1), lips, lips.shape)
rejects('occlusion above normalized range rejected', np.full_like(lips, 2), lips, lips.shape)
rejects('lips above normalized range rejected', occ, np.full_like(lips, 2), lips.shape)
rejects('missing native shape rejected', occ, lips, None)
rejects('zero native dimension rejected', occ, lips, (0, 256))
rejects('nonfinite native dimension rejected', occ, lips, (np.inf, 256))
rejects('short native shape rejected', occ, lips, (256,))

real_settings = get_faceswap_config()
for label, outer, inner in (
    ('even outer', 24, 7), ('outer below inner', 7, 25), ('equal kernels', 7, 7),
    ('zero inner', 25, 0), ('boolean', True, 7), ('float', 25., 7), ('string', '25', 7),
):
    fixture = dict(real_settings, mouth_occlusion_ring_outer_kernel=outer, mouth_occlusion_ring_inner_kernel=inner)
    namespace['get_faceswap_config'] = lambda: fixture
    rejects(f'invalid config {label} rejected', occ, lips, lips.shape)
namespace['get_faceswap_config'] = get_faceswap_config

report = dict(scope='Synthetic mask/config fixtures; actual production helper, baseline arithmetic and gate AST; no models or renders',
              current_sha256=hashlib.sha256(CURRENT.read_bytes()).hexdigest(),
              baseline_sha256=hashlib.sha256(BEFORE.read_bytes()).hexdigest(),
              config_sha256=hashlib.sha256((ROOT / 'config/model_configs.json').read_bytes()).hexdigest(),
              measurements=measurements, checks=checks,
              passed=sum(check['passed'] for check in checks), failed=sum(not check['passed'] for check in checks))
(OUT / 'ring_scale_contract.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(f"Result: {report['passed']} PASS / {report['failed']} FAIL")
raise SystemExit(0 if report['failed'] == 0 else 1)
