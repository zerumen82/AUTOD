"""Download Animate Models from HuggingFace"""
import os, sys, time

TOKEN = os.environ.get("HF_TOKEN", "")
BASE = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"

MODELS = [
    {
        "name": "UMT5 Text Encoder",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors?download=1",
        "dest": os.path.join(BASE, "text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
        "expected_mb": 1100
    },
    {
        "name": "Wan2.2 I2V 14B (fp8)",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors?download=1",
        "dest": os.path.join(BASE, "diffusion_models", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"),
        "expected_mb": 14000
    },
]

def download_with_progress(url, dest, token, label="", expected_mb=0):
    import requests
    if os.path.exists(dest):
        mb = os.path.getsize(dest) / 1024 / 1024
        print(f"[OK] {label}: {mb:.0f}MB already")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[DL] {label} ({expected_mb}MB)...")
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "Mozilla/5.0"}
    r = requests.get(url, stream=True, headers=headers, timeout=30)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    t0 = time.time()
    with open(dest + ".tmp", "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                speed = downloaded / 1024 / 1024 / (time.time() - t0 + 0.01)
                eta = (total - downloaded) / (speed * 1024 * 1024 + 1)
                print(f"\r  {label}: {downloaded//1024//1024}/{total//1024//1024}MB ({pct:.0f}%) @ {speed:.1f}MB/s ETA:{eta:.0f}s", end="", flush=True)
    os.rename(dest + ".tmp", dest)
    print(f"\n[OK] {label}: {os.path.getsize(dest)//1024//1024}MB ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    for m in MODELS:
        try:
            download_with_progress(m["url"], m["dest"], TOKEN, m["name"], m["expected_mb"])
        except Exception as e:
            print(f"[FAIL] {m['name']}: {e}")
    print("\nDone!")