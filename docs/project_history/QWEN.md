# AUTOAUTO - AI Video & Image Editor

## Project Overview

**AUTOAUTO** is a comprehensive AI-powered media processing application built on Python, featuring face swapping, image editing, and video generation capabilities. The project integrates multiple AI engines including ComfyUI, Stable Diffusion, FLUX, and various face processing models.

### Core Technologies

- **Primary Language**: Python 3.x
- **UI Framework**: Gradio (local server on port 7861+)
- **AI Engines**: 
  - ComfyUI (video generation, image editing)
  - Stable Diffusion WebUI (optional, ports 9871-9875)
  - FLUX Fill Pipeline (image editing)
  - ONNX Runtime (face detection/swapping)
- **Deep Learning**: PyTorch, CUDA 12.4, cuDNN
- **Computer Vision**: OpenCV, MediaPipe (468 facial landmarks)

### Main Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Face Swap** | Single/multi-face swapping in images/videos with batch processing | ✅ Production |
| **Image Editor** | Intelligent image editing with automatic technique detection (FLUX + ComfyUI) | ✅ Production |
| **Animate Photo** | Image-to-video generation (SVD Turbo, LTX Video, Zeroscope) | ✅ Production |
| **Batch Processing** | Parallel frame processing (60-70% speedup) | ✅ Production |
| **Live Camera** | Real-time face swap with virtual camera output | ✅ Production |
| **Face Manager** | Face detection, analysis, and management tools | ✅ Production |

---

## Project Structure

```
D:\PROJECTS\AUTOAUTO\
├── ui/                          # Gradio UI and tabs
│   ├── main.py                  # Main application entry point
│   ├── globals.py               # Global UI state
│   ├── tabs/                    # Feature tabs
│   │   ├── faceswap_tab.py      # Face swap interface (4735 lines)
│   │   ├── img_editor_tab.py    # Image Editor interface
│   │   ├── animate_photo_tab.py # Video generation interface
│   │   ├── comfy_launcher.py    # ComfyUI launcher
│   │   ├── sd_launcher.py       # SD WebUI launcher
│   │   ├── livecam_tab.py       # Live camera preview
│   │   ├── facemgr_tab.py       # Face management
│   │   └── settings_tab.py      # Application settings
│   └── tob/                     # Third-party backends
│       ├── ComfyUI/             # ComfyUI installation
│       └── stable-diffusion-webui/
│
├── roop/                        # Core processing engine
│   ├── core.py                  # Main runtime logic
│   ├── globals.py               # Global processing state
│   ├── ProcessMgr.py            # Frame processing manager
│   ├── batch_processor.py       # Parallel batch processing
│   ├── swapper.py               # Face swapping logic
│   ├── face_util.py             # Face detection utilities
│   ├── mouth_detector.py        # MediaPipe mouth detection
│   ├── video_ai_enhancer.py     # Video enhancement
│   ├── comfy_client.py          # ComfyUI API client
│   ├── comfy_workflows.py       # ComfyUI workflow definitions
│   └── img_editor/              # Image editing module
│       ├── img_editor_manager.py # Editor logic
│       ├── flux_client.py       # FLUX model client
│       ├── clothing_segmenter.py # CLIPSeg segmentation
│       └── controlnet_utils.py  # ControlNet utilities
│
├── models/                      # AI models (checkpoints, VAE, etc.)
├── checkpoints/                 # Model checkpoints
├── output/                      # Generated outputs
├── config.yaml                  # Application configuration
├── settings.py                  # Settings manager
├── run.py                       # Application launcher
└── QWEN.md                      # This file
```

---

## Building and Running

### Prerequisites

- **GPU**: NVIDIA with 4GB+ VRAM (8GB+ recommended)
- **CUDA**: 12.4 installed at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`
- **Python**: 3.10+ with virtual environment
- **Dependencies**: See `ui/requirements.txt` and `venv/`

### Quick Start

```bash
# Navigate to project directory
cd D:\PROJECTS\AUTOAUTO

# Activate virtual environment (if not auto-activated)
.\venv\Scripts\activate

# Launch the application
python run.py
```

The application will:
1. Configure CUDA/cuDNN paths and DLL loading
2. Set up temporary directories (D:\.autodeep_temp)
3. Apply Windows file lock patches
4. Load PyTorch and ONNX Runtime
5. Launch Gradio UI at `http://127.0.0.1:7861`

### ComfyUI Integration

ComfyUI runs as a subprocess and is accessible at `http://127.0.0.1:8188`:

```bash
# Manual ComfyUI start (if needed)
python ui/tob/ComfyUI/main.py
```

### Stable Diffusion WebUI (Optional)

SD WebUI runs on ports 9871-9875 and can be launched from the "SD Launcher" tab.

---

## Configuration

### config.yaml

```yaml
provider: cuda                    # cuda, cpu, or directml
max_threads: 4                    # Processing threads
video_quality: 14                 # Output quality (0-51)
output_video_format: mp4          # mp4, avi, mkv
output_image_format: png          # png, jpg, webp
clear_output: true                # Auto-clean temp files
server_port: 7861                 # Gradio port
```

### roop/globals.py Key Settings

```python
# Face Swap Quality
blend_ratio = 0.95                # 95% source face preservation
distance_threshold = 0.35         # Face matching strictness
face_similarity_threshold = 0.2   # Embedding similarity

# Performance
batch_processing_size = 4         # Parallel frames (1-16)
max_batch_threads = 4             # Thread pool size
execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Features
preserve_mouth_expression = True  # MediaPipe mouth detection
use_mediapipe_detector = True     # 468 landmark detection
temporal_smoothing = True         # Video flicker reduction
```

---

## Development Conventions

### Code Style

- **Imports**: Standard library → Third-party → Local modules
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Type Hints**: Used in new code (Python 3.8+ syntax)
- **Error Handling**: Try/except with specific exceptions, logging with `[TAG]` prefix

### Logging Format

```python
print(f"[TAG] Message")  # Console logging
# Common tags: [ENV], [CONFIG], [BATCH], [MOUTH_DETECT], [FLUX], [COMFY]
```

### Testing Practices

- Manual testing via UI tabs
- Debug scripts in root (`check_*.py`, `debug_*.py`)
- Test data in `testdata/` directory

### Git Workflow

- Main development on `main` branch
- Large binaries (models, checkpoints) ignored via `.gitignore`
- Submodules for ComfyUI and SD WebUI

---

## Available Models

### Face Enhancers

| Model | Status | Use Case |
|-------|--------|----------|
| **CodeFormer** | ✅ Default | Best identity preservation (LMD 5.41) |
| **RestoreFormer++** | ✅ Available | High quality alternative |

### Video Generation (ComfyUI)

| Model | VRAM | Description |
|-------|------|-------------|
| **SVD Turbo** | 4-6GB | Fast generation (~1-2s/video) |
| **LTX Video 0.9.1** | 6-7GB | High quality, replaces Wan2.2 |
| **Zeroscope V2 XL** | 4GB | Lightweight prototyping |

### Image Editing

| Model | Use |
|-------|-----|
| **FLUX Fill** | Primary editor engine |
| **ControlNet OpenPose** | Pose manipulation |
| **ControlNet Tile** | Upscaling/enhancement |
| **IP-Adapter** | Identity preservation |
| **CLIPSeg** | Automatic segmentation |

---

## Key Workflows

### Face Swap Pipeline

```
1. User uploads source image(s) and target video/image
2. Face detection (MediaPipe 468 landmarks or InsightFace fallback)
3. Face matching with similarity threshold (0.2)
4. Frame extraction (if video)
5. Batch processing (parallel threads, batch_size=4)
6. Face swap with blending (95% source preservation)
7. Mouth expression preservation (MediaPipe detection)
8. Optional enhancement (CodeFormer/RestoreFormer++)
9. Video reassembly with FFmpeg
10. Output to /output directory
```

**Face Mask Optimization (v2)**: 
- Elliptical mask covers 104% height × 96% width for full forehead coverage
- Center adjusted 8% upward to prevent "fringe/hair line" effect
- Keypoint-based mask includes forehead point for better blending

### Image Editor Pipeline

```
1. User uploads image and writes natural prompt
2. Prompt analysis detects required techniques:
   - "bailando", "pose" → OpenPose
   - "mejora", "calidad" → Tile/Upscale
   - "ropa", "vestido" → CLIPSeg inpainting
3. FLUX Fill generation (primary) or ComfyUI fallback
4. Multiple variations (1-8, default 6)
5. Face preservation swap (optional)
6. Output display and download
```

### Animate Photo Pipeline

```
1. User uploads image and selects model
2. ComfyUI workflow construction:
   - SVD Turbo: Fast, 720x480, 24 frames
   - LTX Video: Quality, 320x192, 25 frames
   - Zeroscope: Lightweight, 576x320, 48 frames
3. Queue submission to ComfyUI
4. Video generation and download
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **CUDA out of memory** | Reduce `batch_processing_size` to 2 or 1 |
| **ComfyUI not connected** | Start ComfyUI from tab or `python ui/tob/ComfyUI/main.py` |
| **No faces detected** | Enable MediaPipe: `use_mediapipe_detector = True` |
| **File lock errors** | Auto-handled by `resilient_move/remove` patches in `run.py` |
| **Gradio port in use** | App auto-finds available port (7861+) |

### Debug Scripts

```bash
# Check available nodes
python check_available_nodes.py

# Check VAE models
python check_available_vaes.py

# Check SVD configuration
python check_svd_conditioning_node.py

# Verify test images
python check_test_images.py
```

---

## Performance Optimization

### VRAM Management

```python
# For 4GB GPU
batch_processing_size = 2
max_batch_threads = 2

# For 8GB GPU
batch_processing_size = 8
max_batch_threads = 4

# For 12GB+ GPU
batch_processing_size = 16
max_batch_threads = 8
```

### Speed Tips

1. Use **batch processing** for videos (60-70% faster)
2. Enable **MediaPipe** for faster face detection
3. Use **SVD Turbo** for quick video generation
4. Set `enhancer_blend_factor = 0.05` for minimal enhancement overhead

---

## API Integration

### ComfyUI Client

```python
from roop.comfy_client import ComfyClient

client = ComfyClient("http://127.0.0.1:8188")
workflow = {...}  # Workflow JSON
result = client.queue_workflow(workflow)
```

### FLUX Client

```python
from roop.img_editor.flux_client import get_flux_client, is_flux_available

if is_flux_available():
    client = get_flux_client()
    result = client.generate(prompt, image, mask)
```

---

## Documentation Files

| File | Description |
|------|-------------|
| `SD_EDITOR_README.md` | Image Editor feature documentation |
| `BATCH_PROCESSING_README.md` | Batch processing configuration |
| `ENHANCERS_2025_README.md` | Face enhancer models guide |
| `MOUTH_DETECTION_README.md` | MediaPipe mouth detection |
| `ComfyUI_Technical_Documentation.md` | ComfyUI technical details |
| `ESTADO_SISTEMA.md` | System status (Spanish) |
| `INFORME_ANALISIS_TABS_2026-02-13.md` | Technical analysis report |

---

## Version Information

- **Current Version**: 2026.1
- **Last Updated**: March 2026
- **Python**: 3.10+
- **Gradio**: 4.44.1+
- **PyTorch**: 2.0.0+
- **CUDA**: 12.4
- **MediaPipe**: 0.10.0+

### Recent Updates

- **Face Mask Optimization v2**: Fixed "fringe/hair line" effect by expanding mask coverage to 104% height, adjusted center 8% upward, and added forehead point to keypoint-based masks
- **Image Editor**: Renamed from "SD Editor" to better reflect FLUX + ComfyUI capabilities
- **Metrics Panel**: Real-time processing metrics with progress, FPS, frames, and status (no overlapping Gradio progress bar)

---

## Contact & Support

For issues or questions:
1. Check ComfyUI is running (`http://127.0.0.1:8188`)
2. Review console logs for `[ERROR]` tags
3. Verify models are installed in correct directories
4. Check `error_log.txt` for detailed errors
