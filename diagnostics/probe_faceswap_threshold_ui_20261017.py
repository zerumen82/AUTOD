"""Independent CPU acceptance matrix and isolated Gradio callbacks; no inference."""
import ast
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from roop.faceswap_config import get_faceswap_config


ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'roop/ProcessMgr.py').read_text(encoding='utf-8'))
SOURCE = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'ProcessMgr')
NAMES = {'_destination_match_threshold', '_calculate_similarity', '_identity_margin',
         '_validate_locked_identity', '_try_strict_same_person_reacquire',
         '_selected_target_is_unique', '_setup_selected_faces_frame_for_video',
         '_as_ref_list', '_select_source_face_by_similarity', '_score_source_face_quality',
         '_source_min_side', '_cluster_source_candidates', '_is_profile_face',
         '_resolve_swap_embedding'}
METHODS = [n for n in SOURCE.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
assert len(METHODS) == len(NAMES)
NS = dict(np=np, cv2=cv2, os=os, get_faceswap_config=get_faceswap_config,
          roop=SimpleNamespace(globals=SimpleNamespace(selected_face_references={})),
          get_face_center=lambda f: ((f.bbox[0]+f.bbox[2])/2, (f.bbox[1]+f.bbox[3])/2))
CLASS = ast.ClassDef(name='ProcessMgr', bases=[], keywords=[], body=METHODS, decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[CLASS], type_ignores=[])),
             '<real production bodies>', 'exec'), NS)
MANAGER = NS['ProcessMgr']


def face(sim, x=0):
    return SimpleNamespace(embedding=np.array([sim, np.sqrt(1-sim**2)], np.float32),
                           bbox=np.array([x, 0, x+100, 120]), kps=None, det_score=.95)


def manager(value):
    item = MANAGER()
    item.options = SimpleNamespace(face_distance_threshold=value)
    return item


class ThresholdMatrix(unittest.TestCase):
    def test_default_preserves_every_existing_gate_exactly(self):
        self.assertEqual(get_faceswap_config()['matching_distance_default'], .20)
        for value in (.20, None, 'invalid', float('nan'), float('inf')):
            item = manager(value)
            for base in (0., .195, .20, .22, .255, .26, .275, .28, .30,
                         .32, .34, .35, .36, .38, .42, .48, .50, 1.):
                self.assertEqual(item._destination_match_threshold(base), base)
        self.assertEqual(MANAGER()._destination_match_threshold(.35), .35)

    def test_slider_direction_and_clamping(self):
        for value, expected in ((.01, .54), (.10, .45), (.20, .35), (.40, .15), (1., 0.),
                                (-10., .55), (10., 0.)):
            self.assertAlmostEqual(manager(value)._destination_match_threshold(.35), expected)

    def test_locked_acceptance_frontal_profile_and_recovery(self):
        ref = np.array([1., 0.], np.float32)
        for profile in (False, True):
            for recovery in (False, True):
                for multi in (False, True):
                    base = (.36 if recovery and not multi else .42) * (.85 if profile else 1.)
                    for slider in (.01, .20, .40):
                        item = manager(slider)
                        for sim in (.15, .25, .35, .45, .65):
                            candidate = face(sim)
                            others = [candidate, face(.02, 150)] if multi else [candidate]
                            actual, _ = item._validate_locked_identity(
                                others, ref, candidate, multi_face=multi,
                                closeup_recovery=recovery, is_profile=profile)
                            gate = np.clip(base + .20 - slider, 0., 1.)
                            retry = np.clip(base - .08 + .20 - slider, 0., 1.)
                            expected = sim >= gate or (multi and sim >= retry)
                            self.assertEqual(actual, expected, (profile, recovery, multi, slider, sim))

    def test_ambiguity_stays_rejected_when_permissive(self):
        item = manager(1.)
        ref = np.array([1., 0.], np.float32)
        a, b = face(.8), face(.79, 150)
        self.assertFalse(item._validate_locked_identity([a, b], ref, a, multi_face=True)[0])
        item._bank_best_sim = lambda emb, name: float(emb[0])
        self.assertFalse(item._selected_target_is_unique(a, [a, b], 'v.mp4'))

    def test_strict_reacquire_changes_acceptance_preserves_position(self):
        ref, anchor, candidate = np.array([1., 0.]), face(1.), face(.45)
        for slider, accepted in ((.01, False), (.20, False), (.40, True)):
            item = manager(slider)
            item._bank_best_sim = lambda *a: None
            with contextlib.redirect_stdout(io.StringIO()):
                result = item._try_strict_same_person_reacquire([candidate], ref, anchor, 'v.mp4', 2)
            self.assertEqual(result is candidate, accepted)
            self.assertIsNone(item._try_strict_same_person_reacquire([face(.99, 1000)], ref, anchor, 'v.mp4', 2))

    def test_setup_default_and_strict_single_face_fallback(self):
        selected, candidate = face(1.), face(.1)
        refs = [{'face_obj': selected, 'embedding': selected.embedding, 'bbox': selected.bbox}]
        for slider, accepted in ((.01, False), (.20, True), (.40, True)):
            item = manager(slider)
            item._video_reference_bank = lambda *a: refs
            with contextlib.redirect_stdout(io.StringIO()):
                item._setup_selected_faces_frame_for_video('v.mp4', [candidate])
            self.assertEqual(getattr(item, '_target_face_assigned_v.mp4', None) is candidate, accepted)

    def test_source_choice_scores_and_embedding_are_invariant(self):
        target, a, b = face(1.), face(.95), face(.1)
        for slider in (.01, .20, .40, 1.):
            item = manager(slider)
            with contextlib.redirect_stdout(io.StringIO()):
                for quality_first in (False, True):
                    self.assertIs(item._select_source_face_by_similarity(
                        target, [a, b], quality_first=quality_first), a)
            self.assertAlmostEqual(item._calculate_similarity(target.embedding, a.embedding), .95, places=5)
            np.testing.assert_allclose(item._resolve_swap_embedding(a, target).embedding,
                                       a.embedding / np.linalg.norm(a.embedding), atol=1e-7)


class UIControls(unittest.TestCase):
    def test_real_gradio_config_and_start_callback(self):
        import gradio as gr
        from fastapi.testclient import TestClient
        from gradio.routes import App
        import roop.globals as globals_
        from ui.tabs.faceswap.ui import build_faceswap_ui
        from ui.tabs.faceswap.events import wire_events, on_start_process
        with gr.Blocks() as demo:
            controls = build_faceswap_ui()
            wire_events(controls)
        self.assertNotIn('fake_preview', controls)
        self.assertEqual(controls['face_distance'].value, .20)
        with TestClient(App.create_app(demo)) as client:
            response = client.get('/config')
        self.assertEqual(response.status_code, 200)
        fn = next(fn for fn in demo.fns.values() if fn.fn is on_start_process)
        self.assertEqual(len(fn.inputs), 7)
        self.assertEqual([c._id for c in fn.inputs], [controls[k]._id for k in
            ('autorotate', 'smoothing', 'face_distance', 'blend_ratio',
             'enhancer_blend', 'enhancer', 'source_random')])
        with patch('ui.tabs.faceswap.logic.start_swap', return_value=iter([(50, 'CPU fixture')])) as start:
            values = list(fn.fn(False, False, .10, .75, .60, 'GFPGAN', False))
        self.assertEqual(values, [(50, 'CPU fixture')])
        self.assertEqual(globals_.distance_threshold, .10)
        self.assertFalse(globals_.autorotate_faces)
        self.assertFalse(globals_.temporal_smoothing)
        self.assertEqual(start.call_args.kwargs['face_distance'], .10)
        self.assertFalse(start.call_args.kwargs['autorotate'])
        self.assertFalse(start.call_args.kwargs['temporal_smoothing'])


if __name__ == '__main__':
    unittest.main()
