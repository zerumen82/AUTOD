"""Compare real CPU composition against a saved ProcessMgr source snapshot.

Both variants use current local alignment, model weights and measured faces.
This tests composition in photographs; it does not certify identity or videos.
No downloads, GPU sessions, UI startup or production edits are performed.
"""

import argparse
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import hashlib
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
import json
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

# Set before importing anything that can import PyTorch.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import insightface
import numpy as np
import onnxruntime as ort
import torch

import roop.face_util as face_util
import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.processors.FaceSwap import FaceWarpEngine
from roop.types import FaceSet


def digest_array(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def face_record(face, crop):
    return {
        "bbox": np.asarray(face.bbox).tolist(),
        "kps": np.asarray(face.kps).tolist(),
        "embedding_hash": digest_array(face.embedding),
        "crop_hash": digest_array(crop),
        "crop_shape": list(crop.shape),
        "det_score": float(face.det_score),
    }


def write_image(path, pixels):
    assert pixels.size > 0, f"Empty image: {path}"
    assert cv2.imwrite(str(path), pixels), f"Unable to save {path}"


@contextmanager
def persist_metadata(metadata, output):
    try:
        yield
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


@contextmanager
def cpu_sessions_only():
    """Force CPU even for INSwapper's temporary session without providers.

    These remain real ONNX sessions. Explicit GPU requests fail rather than
    silently falling back. InsightFace is imported before installing the
    wrapper so its existing PickableInferenceSession subclass stays valid.
    """
    original = ort.InferenceSession
    sessions = []

    class CpuSession(original):
        def __init__(self, path_or_bytes, sess_options=None, providers=None,
                     provider_options=None, **kwargs):
            assert providers is None or providers == ["CPUExecutionProvider"], providers
            super().__init__(path_or_bytes, sess_options=sess_options,
                             providers=["CPUExecutionProvider"],
                             provider_options=provider_options, **kwargs)
            assert self.get_providers() == ["CPUExecutionProvider"]
            self.diagnostic_path = str(path_or_bytes)
            self.diagnostic_calls = 0
            sessions.append(self)

        def run(self, *args, **kwargs):
            self.diagnostic_calls += 1
            return super().run(*args, **kwargs)

        def run_with_iobinding(self, *args, **kwargs):
            self.diagnostic_calls += 1
            return super().run_with_iobinding(*args, **kwargs)

    ort.InferenceSession = CpuSession
    try:
        yield sessions
    finally:
        ort.InferenceSession = original


def load_baseline(path):
    loader = SourceFileLoader(f"composition_baseline_{time.time_ns()}", str(path))
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module.ProcessMgr


def verify_processors(manager):
    engine = manager.processors["faceswap"]
    masker = manager.processors["mask_xseg"]
    enhancer = manager.processors["enhance_gfpgan"]
    assert not engine.is_256 and not engine.source_is_image, engine.model_path
    for processor, session in (
        (engine, engine.model.session),
        (masker, masker.model_xseg),
        (enhancer, enhancer.model_gfpgan),
    ):
        assert processor.devicename == "cpu", processor.devicename
        assert session is not None, "Required local processor did not load"
        assert session.get_providers() == ["CPUExecutionProvider"]
    return engine, masker, enhancer


def run_variant(manager_class, source, target, frame, photo, output, label,
                blend, enhancer_blend):
    manager_class._swap_call_count = 0
    manager = manager_class()
    stages = {"faceswap": 0, "xseg": 0, "gfpgan": 0}
    swaps = []
    target_copy = deepcopy(target)
    source_copy = deepcopy(source)
    coordinates_before = {
        "bbox": np.asarray(target_copy.bbox).copy(),
        "kps": np.asarray(target_copy.kps).copy(),
    }
    try:
        manager.initialize(
            [FaceSet([source_copy])], [target_copy],
            SimpleNamespace(blend_ratio=blend, current_video_path=None),
            face_swap_mode="selected_faces",
        )
        manager._is_single_photo = True
        assert len(manager.input_facesets) == 1
        assert len(manager.source_embeddings_cache) == 1
        engine, masker, enhancer = verify_processors(manager)
        real_engine_run, real_mask_run, real_enhancer_run = engine.Run, masker.Run, enhancer.Run

        def measured_swap(_engine, swap_source, swap_target, image, paste_back=True):
            stages["faceswap"] += 1
            result = real_engine_run(swap_source, swap_target, image, paste_back=paste_back)
            if result is not None:
                assert isinstance(result, tuple) and len(result) == 2
                pixels, matrix = result
                record = {
                    "source_embedding_hash": digest_array(swap_source.embedding),
                    "target_bbox": np.asarray(swap_target.bbox).tolist(),
                    "target_kps": np.asarray(swap_target.kps).tolist(),
                    "matrix": np.asarray(matrix).tolist(),
                    "raw_hash": digest_array(pixels),
                }
                swaps.append(record)
                write_image(output / f"{label}_raw_{len(swaps)}.png", pixels)
            return result

        def measured_mask(_masker, *args, **kwargs):
            stages["xseg"] += 1
            mask = real_mask_run(*args, **kwargs)
            write_image(output / f"{label}_xseg_{stages['xseg']}.png",
                        (np.clip(mask, 0, 1) * 255).astype(np.uint8))
            return mask

        def measured_enhancer(_enhancer, *args, **kwargs):
            stages["gfpgan"] += 1
            return real_enhancer_run(*args, **kwargs)

        engine.Run = MethodType(measured_swap, engine)
        masker.Run = MethodType(measured_mask, masker)
        enhancer.Run = MethodType(measured_enhancer, enhancer)
        started = time.monotonic()
        result = manager._process_face_swap_v21(
            source_copy, target_copy, frame.copy(), frame,
            enable_temporal_smoothing=False, photo_path=str(photo),
        )
        seconds = time.monotonic() - started
        assert result is not None and result.shape == frame.shape
        coordinates_unchanged = all(
            np.array_equal(value, np.asarray(getattr(target_copy, name)))
            for name, value in coordinates_before.items()
        )
        assert coordinates_unchanged, "Manager mutated measured target coordinates"
        diff = cv2.absdiff(result, frame).max(axis=2)
        write_image(output / f"{label}_full.png", result)
        write_image(output / f"{label}_changed.png", np.where(diff > 0, 255, 0).astype(np.uint8))
        model_calls = {
            "faceswap": engine.model.session.diagnostic_calls,
            "xseg": masker.model_xseg.diagnostic_calls,
            "gfpgan": enhancer.model_gfpgan.diagnostic_calls,
        }
        successful = bool(swaps) and model_calls["faceswap"] > 0 and bool((diff > 0).any())
        assert model_calls["xseg"] > 0 or not successful, "Changed result without real XSeg inference"
        metadata = {
            "label": label, "seconds": seconds, "stage_calls": stages,
            "model_calls": model_calls, "swaps": swaps,
            "changed_pixels": int((diff > 0).sum()),
            "changed_fraction_gt10": float((diff > 10).mean()),
            "coordinates_unchanged": coordinates_unchanged,
            "source_facesets": len(manager.input_facesets),
            "source_embeddings_cached": len(manager.source_embeddings_cache),
            "blend": blend, "enhancer_blend": enhancer_blend,
            "successful": successful,
        }
        return result, metadata
    finally:
        # Release processors, not ProcessMgr.Release (which also cleans temps).
        for processor in manager.processors.values():
            processor.Release()
        manager.processors.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("targets", type=Path, nargs="+")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--enhancer-blend", type=float, default=0.70)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert not torch.cuda.is_available(), "Torch still sees CUDA"
    required = [
        args.source, *args.targets, args.baseline,
        root / "models/inswapper_128_facefusion.onnx",
        root / "models/xseg.onnx", root / "models/GFPGANv1.4.onnx",
        *(root / "models/buffalo_l" / name for name in (
            "det_10g.onnx", "w600k_r50.onnx", "1k3d68.onnx",
            "2d106det.onnx", "genderage.onnx",
        )),
    ]
    for path in required:
        if not path.is_file():
            parser.error(f"Required local file is missing; no download attempted: {path}")
    assert args.source_index >= 0 and args.target_index >= 0
    assert 0 <= args.blend <= 1 and 0 <= args.enhancer_blend <= 1
    output = args.report / f"composition_cpu_{time.time_ns()}"
    output.mkdir(parents=True)
    metadata = {
        "source": str(args.source.resolve()), "cpu_only": True,
        "insightface_version": insightface.__version__,
        "completed": False,
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "current_sha256": hashlib.sha256((root / "roop/ProcessMgr.py").read_bytes()).hexdigest(),
        "alignment": "Current FaceWarpEngine for both variants; no synthetic landmarks",
        "quality_helpers": "Current quality_enhancements shared by both variants",
        "cases": [],
    }
    print(output, flush=True)
    with persist_metadata(metadata, output), (output / "run.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        with cpu_sessions_only():
            analyser = face_util.get_face_analyser_cpu()
            assert analyser is not None
            for model in analyser.models.values():
                assert model.session.get_providers() == ["CPUExecutionProvider"]
            face_util.FACE_ANALYSER = analyser
            roop.globals.execution_providers = ["CPUExecutionProvider"]
            roop.globals.selected_enhancer = "GFPGAN"
            roop.globals.use_enhancer = True
            roop.globals.enhancer_blend_factor = args.enhancer_blend
            source_faces = face_util.extract_face_images(str(args.source), is_source_face=True)
            assert len(source_faces) > args.source_index, "Real source extraction failed"
            source, source_crop = source_faces[args.source_index]
            metadata["source_detected_count"] = len(source_faces)
            metadata["source_face"] = face_record(source, source_crop)
            write_image(output / "source_crop.png", source_crop)
            baseline_class = load_baseline(args.baseline)
            for index, photo in enumerate(args.targets):
                case_output = output / f"case_{index}_{photo.stem}"
                case_output.mkdir()
                frame = cv2.imdecode(np.fromfile(photo, dtype=np.uint8), cv2.IMREAD_COLOR)
                assert frame is not None, photo
                write_image(case_output / "original.png", frame)
                faces = face_util.extract_face_images(str(photo))
                case = {"target": str(photo.resolve()), "detected_count": len(faces)}
                metadata["cases"].append(case)
                if len(faces) <= args.target_index:
                    case["skip_reason"] = "Target extraction returned no selected measured face"
                    continue
                target, target_crop = faces[args.target_index]
                case["target_face"] = face_record(target, target_crop)
                write_image(case_output / "target_crop.png", target_crop)
                points = np.asarray(target.kps, dtype=np.float32)
                eye_relative = float(np.linalg.norm(points[1] - points[0])) / float(target.bbox[2] - target.bbox[0])
                case["eye_relative"] = eye_relative
                if not 0.05 < eye_relative < 1.20:
                    case["skip_reason"] = "Selected real face has degenerate kps; no forcing/bypass"
                    continue
                expected_matrix, how = FaceWarpEngine.estimate_constrained_norm(
                    frame.shape[:2], target.kps, 128, bbox=target.bbox,
                )
                case["alignment_method"] = how
                case["expected_matrix"] = expected_matrix.tolist()
                renders = []
                for label, manager_class in (("before", baseline_class), ("current", ProcessMgr)):
                    result, measurement = run_variant(
                        manager_class, source, target, frame, photo, case_output,
                        label, args.blend, args.enhancer_blend,
                    )
                    renders.append(result)
                    case[label] = measurement
                case["same_raw_swap"] = case["before"]["swaps"] == case["current"]["swaps"]
                assert case["same_raw_swap"], "Comparison changed geometry, source or raw model output"
                assert face_record(source, source_crop) == metadata["source_face"]
                assert face_record(target, target_crop) == case["target_face"]
                case["before_vs_current_changed_pixels"] = int((cv2.absdiff(*renders).max(axis=2) > 0).sum())
                write_image(case_output / "compare_full.png", np.hstack((frame, *renders)))
                bbox = np.asarray(target.bbox, np.float64)
                span = bbox[2:] - bbox[:2]
                lo = np.maximum(0, np.floor(bbox[:2] - span / 2)).astype(int)
                hi = np.minimum(np.array([frame.shape[1], frame.shape[0]]), np.ceil(bbox[2:] + span / 2)).astype(int)
                write_image(case_output / "compare_crop.png", np.hstack(tuple(
                    image[lo[1]:hi[1], lo[0]:hi[0]] for image in (frame, *renders)
                )))
                print(json.dumps(case), flush=True)
        metadata["completed"] = True
    print(output, flush=True)
    assert all("skip_reason" not in case and case["current"]["successful"] for case in metadata["cases"]), "One or more real compositions skipped/unchanged; inspect evidence"


if __name__ == "__main__":
    main()
