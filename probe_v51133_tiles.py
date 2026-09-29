import numpy as np
import sys

sys.path.insert(0, ".")

from roop.processors.Frame_Upscale import Frame_Upscale


def fake_upscale2x(img):
    """Upscale x2 por nearest (sin modelo): contenido idéntico al original escalado."""
    return np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)


def run_pipeline(img, size):
    fu = Frame_Upscale()
    fu.scale = 2
    tiles, pad_w, pad_h = fu.create_tile_frames(img, size)
    up_tiles = [fake_upscale2x(t) for t in tiles]
    merged = fu.merge_tile_frames(
        up_tiles,
        img.shape[1] * 2, img.shape[0] * 2,
        pad_w * 2, pad_h * 2,
        (size[0] * 2, size[1] * 2, size[2] * 2),
    )
    return merged


def check(name, img, size, ref2x):
    out = run_pipeline(img, size)
    ok_shape = out.shape == ref2x.shape
    # 1) Sin píxeles negros (la rejilla del bug era 0/eps = 0)
    dark = float((out.min(axis=2) < 3).mean())
    # 2) Alineación: diferencia media vs referencia (nearest) en zonas sin gradiente fuerte
    diff = np.abs(out.astype(np.int16) - ref2x.astype(np.int16)).mean()
    # 3) Sin rejilla: comparar filas/columnas en antiguas fronteras de tile (cada core px)
    core = size[0] - 2 * size[2]
    border_rows = list(range(core * 2 - 3, core * 2 + 3))
    row_jumps = np.abs(np.diff(out.astype(np.int16), axis=0)).max(axis=(1, 2))
    max_jump_at_border = float(row_jumps[border_rows].max())
    global_max_jump = float(row_jumps.max())
    ok = ok_shape and dark == 0.0 and diff < 3.0 and max_jump_at_border <= global_max_jump + 1e-6
    print(f"[{name}] shape_ok={ok_shape} dark_frac={dark:.5f} diff_vs_ref={diff:.3f} "
          f"jump@border={max_jump_at_border:.0f} jump_max={global_max_jump:.0f} -> {'OK' if ok else 'FAIL'}")
    return ok


ok = True

# Caso real del log: 771×1024 (H×W), esrganx2 → size=(256,12,10)
img = (np.random.default_rng(7).integers(30, 225, (1024, 771, 3))).astype(np.uint8)
ref = fake_upscale2x(img)
ok &= check("771x1024 esrganx2", img, (256, 12, 10), ref)

# Caso x4 pequeño: size=(128,8,6)
img2 = (np.random.default_rng(3).integers(30, 225, (300, 200, 3))).astype(np.uint8)
fu = Frame_Upscale(); fu.scale = 2  # fake x2 para comparar contra nearest ref
tiles, pw, ph = fu.create_tile_frames(img2, (128, 8, 6))
up = [fake_upscale2x(t) for t in tiles]
out2 = fu.merge_tile_frames(up, img2.shape[1]*2, img2.shape[0]*2, pw*2, ph*2, (128*2, 8*2, 6*2))
ref2 = fake_upscale2x(img2)
d2 = np.abs(out2.astype(np.int16) - ref2.astype(np.int16)).mean()
dark2 = float((out2.min(axis=2) < 3).mean())
ok2 = out2.shape == ref2.shape and dark2 == 0.0 and d2 < 3.0
print(f"[300x200 tile128] shape={out2.shape == ref2.shape} dark={dark2:.5f} diff={d2:.3f} -> {'OK' if ok2 else 'FAIL'}")
ok &= ok2

# Imagen con gradiente suave (peor caso para costuras visibles)
gx = np.linspace(40, 210, 771, dtype=np.float32)
gy = np.linspace(60, 190, 1024, dtype=np.float32)
grad = (gy[:, None] * np.ones(771)[None, :] + gx[None, :] * 0.4)[..., None] * np.ones(3)[None, None]
img3 = grad.clip(0, 255).astype(np.uint8)
ok &= check("gradiente 771x1024", img3, (256, 12, 10), fake_upscale2x(img3))

print("TOTAL:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
