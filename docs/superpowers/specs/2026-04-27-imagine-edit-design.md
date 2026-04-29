# Design Spec: Imagine Edit - Grok-Style Image Editor

## 1. Vision & Goals
Transform the current technical Image Editor into a "Prompt-First" experience similar to Grok Imagine. The system should automatically resolve technical parameters (denoise, guidance, engine) based on natural language analysis, minimizing user friction while maximizing quality.

## 2. User Experience (UI)
- **Main Interface**: A clean, dark-themed canvas with a prominent floating prompt bar at the bottom.
- **Smart Tags**: Predefined chips (e.g., "Change Clothing", "Expand Background", "Ultra Quality") that append context to the prompt and trigger specific backend behaviors.
- **AI Feedback**: A status badge (e.g., "✨ AI ANALYZER: READY") that gives the user confidence that the system is "thinking".
- **Advanced Mode**: A hidden settings panel (accessible via a gear icon) for manual overrides of steps, guidance, and denoise.

## 3. Backend Intelligence (The "Brain")
### A. Parameter Resolution
The system will use the `PromptRewriter` and `PromptAnalyzer` to determine:
- **Change Intensity**: 
    - `Low`: Subtle retouching (Denoise ~0.3 - 0.5)
    - `Medium`: Fashion/Clothing changes (Denoise ~0.6 - 0.75)
    - `High`: Pose changes or full regeneration (Denoise ~0.8 - 0.95)
- **Engine Selection**:
    - **FLUX.2-klein**: Default for high-quality edits.
    - **FLUX.1-schnell**: For fast previews.
    - **SDXL/ControlNet**: For specific structural tasks (tiling/upscaling).

### B. Automatic Workflow
1. **User enters prompt** (e.g., "ponle un vestido rojo").
2. **AI Rewriter** expands this into a detailed SD prompt.
3. **AI Analyzer** detects this as a "Medium" intensity clothing change.
4. **Manager** selects FLUX-klein with `denoise=0.7` and `guidance=2.5`.
5. **Execution** happens without the user ever seeing these numbers.

## 4. Technical Architecture
- **UI**: Gradio-based, refactoring `ui/tabs/img_editor_tab.py`.
- **Logic**: Centralizing logic in `roop/img_editor/img_editor_manager.py` using a new `resolve_params` function.
- **Face Preservation**: Automatically enabled for any "Human" related prompts.

## 5. Success Criteria
- Generation starts with a single click after typing a prompt.
- Result quality is equal or better than manual slider adjustment.
- UI feels modern and responsive (less than 5 main controls visible by default).
