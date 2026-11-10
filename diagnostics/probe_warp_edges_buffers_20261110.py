"""Probe: equivalencia bit a bit y pico de memoria de los fundidos por frame (v5.1326).

Compara las implementaciones anteriores (c3/s3 de 3 canales + astype por frame)
de `_prime_warp_edges` y `_soothe_swap_ring` con las nuevas (búferes in-place y
mezcla por canal) sobre datos aleatorios. No toca modelos ni GPU.

Uso: venv/Scripts/python.exe diagnostics/probe_warp_edges_buffers_20261110.py
"""
import sys
import tracemalloc

import cv2
import numpy as np


# ---------------------------------------------------------------- prime (viejo)
def prime_old(warped_face, original_frame, mask, face_w, face_h):
    m = np.clip(mask.astype(np.float32), 0, 1)
    edge = np.clip((0.72 - m) / 0.70, 0, 1).astype(np.float32)
    blur = max(11, (min(face_w, face_h) // 10) | 1)
    edge = cv2.GaussianBlur(edge, (blur, blur), 0) * 0.80
    dark_map = np.max(warped_face.astype(np.float32), axis=2)
    dark_threshold = 15.0
    dark_mask = np.clip((dark_threshold - dark_map) / dark_threshold, 0, 1).astype(np.float32)
    dark_mask = cv2.GaussianBlur(dark_mask, ((blur * 2) | 1, (blur * 2) | 1), 0)
    combined = np.maximum(edge, dark_mask)
    if float(combined.max()) < 0.02:
        return warped_face
    c3 = cv2.merge([combined, combined, combined])
    out = warped_face.astype(np.float32) * (1.0 - c3) + original_frame.astype(np.float32) * c3
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- prime (nuevo)
def prime_new(warped_face, original_frame, mask, face_w, face_h):
    m = mask.astype(np.float32, copy=True)
    np.clip(m, 0, 1, out=m)
    edge = np.subtract(np.float32(0.72), m)
    edge /= np.float32(0.70)
    np.clip(edge, 0, 1, out=edge)
    del m
    blur = max(11, (min(face_w, face_h) // 10) | 1)
    edge = cv2.GaussianBlur(edge, (blur, blur), 0)
    edge *= np.float32(0.80)
    dark_map = warped_face.max(axis=2).astype(np.float32)
    dark_threshold = 15.0
    dark_mask = np.subtract(np.float32(dark_threshold), dark_map)
    dark_mask /= np.float32(dark_threshold)
    np.clip(dark_mask, 0, 1, out=dark_mask)
    del dark_map
    dark_mask = cv2.GaussianBlur(dark_mask, ((blur * 2) | 1, (blur * 2) | 1), 0)
    np.maximum(edge, dark_mask, out=edge)
    del dark_mask
    if float(edge.max()) < 0.02:
        return warped_face
    inv = np.subtract(np.float32(1.0), edge)
    out = warped_face.copy()
    for ch in range(3):
        channel = warped_face[:, :, ch].astype(np.float32)
        channel *= inv
        channel += original_frame[:, :, ch].astype(np.float32) * edge
        np.clip(channel, 0, 255, out=channel)
        out[:, :, ch] = channel.astype(np.uint8)
    return out


# ---------------------------------------------------------------- soothe (viejo)
def soothe_old(result_frame, original_frame, mask, face_w, face_h):
    m = np.clip(mask.astype(np.float32), 0, 1)
    if float(m.max()) < 0.05:
        return result_frame
    soothe = np.clip((0.97 - m) / 0.94, 0, 1).astype(np.float32)
    soothe *= (m > 0.01).astype(np.float32) * (m < 0.90).astype(np.float32)
    blur = max(17, (min(face_w, face_h) // 6) | 1)
    soothe = cv2.GaussianBlur(soothe, (blur, blur), 0)
    if float(soothe.max()) < 0.02:
        return result_frame
    soothe *= 0.95
    s3 = cv2.merge([soothe, soothe, soothe])
    blended = (
        result_frame.astype(np.float32) * (1.0 - s3)
        + original_frame.astype(np.float32) * s3
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- soothe (nuevo)
def soothe_new(result_frame, original_frame, mask, face_w, face_h):
    m = mask.astype(np.float32, copy=True)
    np.clip(m, 0, 1, out=m)
    if float(m.max()) < 0.05:
        return result_frame
    soothe = np.subtract(np.float32(0.97), m)
    soothe /= np.float32(0.94)
    np.clip(soothe, 0, 1, out=soothe)
    soothe *= np.logical_and(m > 0.01, m < 0.90)
    blur = max(17, (min(face_w, face_h) // 6) | 1)
    soothe = cv2.GaussianBlur(soothe, (blur, blur), 0)
    if float(soothe.max()) < 0.02:
        return result_frame
    soothe *= np.float32(0.95)
    inv = np.subtract(np.float32(1.0), soothe)
    blended = result_frame.copy()
    for ch in range(3):
        channel = result_frame[:, :, ch].astype(np.float32)
        channel *= inv
        channel += original_frame[:, :, ch].astype(np.float32) * soothe
        np.clip(channel, 0, 255, out=channel)
        blended[:, :, ch] = channel.astype(np.uint8)
    return blended


def peak_mb(fn, *args):
    tracemalloc.start()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def make_case(seed, size, mask_mode):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    if mask_mode == "random":
        mask = rng.random((size, size)).astype(np.float32)
    elif mask_mode == "soft":
        mask = np.zeros((size, size), np.float32)
        cv2.circle(mask, (size // 2, size // 2), size // 3, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
    elif mask_mode == "soft_low":  # pico máximo ~0,6: activa la ruta sin salida temprana
        mask = np.zeros((size, size), np.float32)
        cv2.circle(mask, (size // 2, size // 2), size // 3, 0.6, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
    elif mask_mode == "full":
        mask = np.ones((size, size), np.float32)
    else:
        mask = np.zeros((size, size), np.float32)
    return a, b, mask, size


def run_set(name, old_fn, new_fn, cases):
    fails = 0
    print(f"\n== {name} ==")
    for seed, size, mode in cases:
        a, b, mask, fw = make_case(seed, size, mode)
        expected = old_fn(a, b, mask, fw, fw)
        got = new_fn(a, b, mask, fw, fw)
        equal = np.array_equal(expected, got)
        old_peak = peak_mb(old_fn, a, b, mask, fw, fw)
        new_peak = peak_mb(new_fn, a, b, mask, fw, fw)
        if equal:
            print(
                f"  [PASS] seed={seed} {size}px {mode}: identico | "
                f"pico old={old_peak:.2f} MB new={new_peak:.2f} MB"
            )
        else:
            fails += 1
            print(
                f"  [FAIL] seed={seed} {size}px {mode}: "
                f"{int(np.count_nonzero(expected != got))} px distintos"
            )
    return fails


def main():
    prime_cases = [
        (0, 720, "random"),
        (1, 720, "soft"),
        (2, 720, "full"),
        (3, 720, "empty"),
        (4, 480, "random"),
        (5, 256, "soft"),
    ]
    soothe_cases = [
        (10, 720, "soft"),
        (11, 720, "soft_low"),
        (12, 720, "random"),
        (13, 720, "empty"),
        (14, 480, "soft"),
        (15, 256, "random"),
    ]
    fails = run_set("_prime_warp_edges", prime_old, prime_new, prime_cases)
    fails += run_set("_soothe_swap_ring", soothe_old, soothe_new, soothe_cases)
    print(f"\nTOTAL FAIL={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
