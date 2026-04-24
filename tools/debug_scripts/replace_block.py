#!/usr/bin/env python
"""Replace elif flux block body with correctly indented version"""

with open('roop/img_editor/img_editor_manager.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Generate new block (lines 257 to 311 inclusive)
base = ' ' * 18   # body indent = 18
four = ' ' * 22   # inside if/for/try
eight = ' ' * 26
twelve = ' ' * 30

block = [
    base + "from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client\n",
    base + "ref_meta = kwargs.get('ref_metadata', {})\n",
    base + "guidance = ref_meta.get('guidance_scale', 3.0)\n",
    base + "denoise = ref_meta.get('denoise', 0.8)\n",
    base + "steps = kwargs.get('num_inference_steps', 8)\n",
    base + 'resolution = ref_meta.get(\'resolution_label\', None)  # "512p", "720p", "1024p"\n',
    base + "\n",
    base + "# Detectar si se quiere preservar fondo/composición (múltiples sinónimos en español/inglés)\n",
    base + "prompt_lw = prompt.lower()\n",
    base + "preserve_keywords = [\n",
    four + "'mantener fondo', 'conservar fondo', 'preservar fondo',\n",
    four + "'mantener composicion', 'conservar composicion', 'preservar composicion',\n",
    four + "'mantener composición', 'conservar composición', 'preservar composición',\n",
    four + "'mantener entorno', 'conservar entorno', 'preservar entorno',\n",
    four + "'keep background', 'maintain background', 'preserve background', 'background unchanged',\n",
    four + "'keep composition', 'maintain composition', 'preserve composition',\n",
    four + "'keep scene', 'maintain scene', 'preserve scene',\n",
    four + "'no cambies el fondo', 'do not change background',\n",
    four + "'manteniendo', 'conservando', 'preservando',  # gerundios\n",
    four + "'mantener', 'conservar', 'preservar'  # infinitivos\n",
    base + "]\n",
    base + "preserve_background = any(kw in prompt_lw for kw in preserve_keywords)\n",
    base + "\n",
    base + "client = get_flux_edit_comfy_client()\n",
    base + "client.load(flux_version=engine)\n",
    base + "\n",
    base + "# Advertencia para FLUX.1-schnell (GGUF T5 lento)\n",
    base + "if engine == \"flux_schnell\":\n",
    four + 'print("[WARN] FLUX.1-schnell usa GGUF T5 que es MUY LENTO (~20+ min). Se recomienda FLUX.2-klein.")\n',
    base + "\n",
    base + "# Convertir resolución label a dimensiones\n",
    base + "target_w, target_h = None, None\n",
    base + "if resolution:\n",
    four + "try:\n",
    eight + '# "512p" -> max_side = 512\n',
    eight + "max_side = int(resolution.replace('p', ''))\n",
    eight + "# Calcular dimensiones manteniendo aspect ratio\n",
    eight + "w, h = image.size\n",
    eight + "if w >= h:\n",
    twelve + "target_w = max_side\n",
    twelve + "target_h = int(h * (max_side / w))\n",
    eight + "else:\n",
    twelve + "target_h = max_side\n",
    twelve + "target_w = int(w * (max_side / h))\n",
    eight + "# Ajustar a múltiplo de 16\n",
    eight + "target_w = (target_w // 16) * 16\n",
    eight + "target_h = (target_h // 16) * 16\n",
    eight + 'print(f"[ImgEditor] Resolución objetivo: {target_w}x{target_h} (original {w}x{h})")\n',
    four + "except:\n",
    eight + "pass\n",
    base + "\n",
    base + "res, msg = client.generate(\n",
    four + "image=image, prompt=rewritten_prompt, seed=seed,\n",
    four + "guidance_scale=guidance, denoise=denoise, num_inference_steps=steps,\n",
    four + "flux_version=engine,\n",
    four + "preserve_background=preserve_background,\n",
    four + "target_width=target_w,\n",
    four + "target_height=target_h\n",
    base + ")\n",
    base + "if res and res.image:\n",
    four + "final = res.image\n",
    four + "if face_preserve: final = self._restore_face(image, final)\n",
    four + "return final, f\"FLUX {engine} OK\"\n",
    base + "return None, f\"Error FLUX {engine}: {msg}\"\n",
]

# Replace lines 257-311 (indices 256-310) with block
new_lines = lines[:256] + block + lines[311:]

with open('roop/img_editor/img_editor_manager.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Replaced {len(block)} lines (original 55 lines)")
