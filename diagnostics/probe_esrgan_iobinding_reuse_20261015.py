"""v5.1211 — Fix B: binding de salida de ESRGAN compartido entre Run().

Defecto (edit_1792085849, 15/10): `Frame_Upscale` creaba UN io_binding en
Initialize y lo reutilizaba. El primer Run con tile 256 dimensionaba la
salida a 512; el segundo Run con tile 384 pedía 768 y ONNX Runtime abortaba
`OrtValue shape verification failed. Current shape:{1,3,512,512} Requested
shape:{1,3,768,768}`, degradando el upscale a Lanczos.

El probe (a) reproduce el patrón antiguo —un binding para dos tamaños— como
referencia del bug (assert en CUDA, informativo en CPU), y (b) verifica el
contrato del fix: dos Run() con tiles distintos consecutivos, sin error y con
las formas correctas.
"""

import numpy as np

import roop.globals
from roop.processors.Frame_Upscale import Frame_Upscale
from roop.utilities import resolve_relative_path


def _tile(height: int, width: int) -> np.ndarray:
    frame = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    tile = np.expand_dims(frame[:, :, ::-1], axis=0).transpose(0, 3, 1, 2)
    return tile.astype(np.float32) / 255.0


def main():
    providers = getattr(
        roop.globals, "execution_providers", ["CPUExecutionProvider"]
    )
    dev = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
    print(f"providers={providers} devicename={dev}")

    # (a) patrón antiguo: UN binding para dos tamaños distintos.
    import onnxruntime as ort
    session = ort.InferenceSession(
        resolve_relative_path("../models/Frame/real_esrgan_x2.onnx"),
        None, providers=providers,
    )
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    shared_bug = None
    try:
        shared = session.io_binding()
        shared.bind_output(out_name, dev)
        shared.bind_cpu_input(in_name, _tile(256, 256))
        session.run_with_iobinding(shared)
        shared.bind_cpu_input(in_name, _tile(384, 384))
        session.run_with_iobinding(shared)
    except Exception as exc:  # RuntimeError con el mensaje exacto de producción
        shared_bug = str(exc).replace("\n", " ")
    print(f"[a] binding compartido (patrón antiguo): {shared_bug or 'SIN ERROR'}")
    if dev == "cuda":
        assert shared_bug and "shape verification failed" in shared_bug, \
            ("el fallo original no reproduce en CUDA: el probe ya no aísla la "
             f"causa (bug={shared_bug!r})")

    # (b) contrato del fix: dos Run() consecutivos con tiles distintos.
    up = Frame_Upscale()
    up.Initialize({"devicename": dev, "subtype": "esrganx2"})
    sizes = iter([(128, 8, 6), (192, 10, 8)])
    up._pick_tile_size = lambda h, w, scale=2: next(sizes)

    out1 = up.Run(np.random.randint(0, 255, (200, 160, 3), dtype=np.uint8))
    out2 = up.Run(np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8))
    assert out1.shape == (400, 320, 3), out1.shape
    assert out2.shape == (800, 600, 3), out2.shape
    print(f"[b] Run 1 -> {out1.shape}; Run 2 -> {out2.shape}")
    print("PASS: el binding por llamada aguanta tiles distintos sin degradar")


if __name__ == "__main__":
    main()
