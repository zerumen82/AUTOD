# Fix FaceSwap and UI Bugs Implementation Plan

I'm using the writing-plans skill to create the implementation plan.

**Goal:** Fix four identified bugs: early return in face loop, incorrect warp ROI, UI crash on empty targets, and diagonal artifacts in swaps.

**Architecture:** 
- Fix logic flow in `roop/ProcessMgr.py` to allow multi-face processing and ensure result is returned after the loop.
- Correct coordinate mapping in `roop/processors/FaceSwap.py`.
- Add defensive checks in Gradio event handlers.
- Sanitize mask edges in aligned space to prevent boundary artifacts.

**Tech Stack:** Python, OpenCV, NumPy, Gradio

---

### Task 1: Fix Multiple Face Processing and Early Return

**Files:**
- Modify: `roop/ProcessMgr.py:431, 507`

- [ ] **Step 1: Update face count limit logic**
Change line 431 to allow multiple faces for `selected` and `selected_faces` modes.

```python
# Old
faces_to_process_limited = faces_to_process[:1] if face_swap_mode in ['selected_faces_frame', 'selected_faces', 'selected'] else faces_to_process
# New
faces_to_process_limited = faces_to_process[:1] if face_swap_mode in ['selected_faces_frame'] else faces_to_process
```

- [ ] **Step 2: Move return statement outside the loop**
The `return result_frame` at line 507 is currently inside the `for` loop. Move it one level out.

```python
# ... inside _process_frame ...
                try:
                    result_frame = self._process_face_swap_v21(
                        source_face, target_face, result_frame, frame, enable_temporal_smoothing
                    )
                except Exception as e:
                    print(f"Error procesando cara {i+1}: {e}")
                    continue
            
            # Move this block and the return statement out of the loop
            # ... (frame hold logic) ...
            return result_frame
```

### Task 2: Fix FaceWarpEngine ROI Calculation

**Files:**
- Modify: `roop/processors/FaceSwap.py:81`

- [ ] **Step 1: Use source triangle bounding box for source ROI**
Calculate the bounding box of `src_tri` instead of reusing `rx, ry, rw, rh` from `tgt_tri`.

```python
            rx_s, ry_s, rw_s, rh_s = cv2.boundingRect(src_tri)
            src_roi = aimg_src[ry_s:ry_s+rh_s, rx_s:rx_s+rw_s]
            
            # Also need to adjust tgt_offset and src_offset to be relative to their respective ROIs
            tgt_offset = np.ascontiguousarray((tgt_tri - [float(rx), float(ry)]).astype(np.float32))
            src_offset = np.ascontiguousarray((src_tri - [float(rx_s), float(ry_s)]).astype(np.float32))
```

### Task 3: Fix UI Crash in Target Selection

**Files:**
- Modify: `ui/tabs/faceswap/events.py:543`

- [ ] **Step 1: Add check for empty list**
Ensure `state.list_files_process` is not empty before accessing index 0.

```python
        if not state.list_files_process:
            return gr.update(), None, gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False), [], [], "📄 Página 1 de 1 (0 caras)", gr.update(interactive=False), gr.update(interactive=False)
            
        first_entry = state.list_files_process[0]
```

### Task 4: Fix Diagonal Visual Artifacts

**Files:**
- Modify: `roop/ProcessMgr.py` in `_process_face_swap_v21`

- [ ] **Step 1: Zero out mask edges in aligned space**
Before warping `combined_face` or `mask_align` back to the frame, ensure the outermost pixels are zero to avoid boundary artifacts.

```python
                # Before: final_mask = cv2.warpAffine(combined_face, M_inv, ...)
                # Add:
                combined_face[0, :] = 0
                combined_face[-1, :] = 0
                combined_face[:, 0] = 0
                combined_face[:, -1] = 0
                # Same for fallback mask_align
```

### Task 5: Verification

- [ ] **Step 1: Verify multi-face processing**
Run a swap on a photo/video with multiple faces and ensure all selected faces are swapped.

- [ ] **Step 2: Verify ROI fix**
Run the manual warp (if applicable) or verify no regressions in alignment.

- [ ] **Step 3: Verify UI stability**
Upload invalid files to Destino and ensure no crash occurs.

- [ ] **Step 4: Verify artifact fix**
Check debug images (04_final_result_f1.png) to ensure the diagonal line is gone.
