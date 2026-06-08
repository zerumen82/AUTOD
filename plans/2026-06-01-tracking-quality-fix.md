# Tracking and Face Swap Quality Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve face tracking responsiveness and swap fidelity in videos by relaxing restrictive clamping and adjusting smoothing factors.

**Architecture:** 
1. Relax `BBOX_CLAMP` from 40px to 80px to allow faster movement tracking.
2. Increase temporal smoothing factors (`f_center`, `kps_ema`) to prioritize current frame detection over history when moving.
3. Adjust `Hybrid Embedding` ratio from 80/20 to 60/40 for better adaptation to face rotations.
4. Increase `Ghost Tracking` max shift to prevent lagging behind during temporary tracking loss.

**Tech Stack:** Python, NumPy, OpenCV, Roop (InsightFace)

---

### Task 1: Relax BBox Clamping and Smoothing Factors

**Files:**
- Modify: `roop/ProcessMgr.py`

- [ ] **Step 1: Increase BBox clamping threshold**
  Change the threshold for sudden center jumps from 40px to 80px to allow faster tracking without clamping.
  
  Modify `roop/ProcessMgr.py:1669`:
  ```python
  if dist > 80: # Aumentado de 40 a 80
      half_w = (x2 - x1) / 2
      half_h = (y2 - y1) / 2
      new_cx = prev_center[0] + dx * 0.7 # Aumentado de 0.5 a 0.7 para ser más responsivo
      new_cy = prev_center[1] + dy * 0.7
  ```

- [ ] **Step 2: Increase Center Smoothing Responsiveness**
  Adjust `f_center` to favor the current frame more, especially during movement.
  
  Modify `roop/ProcessMgr.py:1592`:
  ```python
  f_center = 0.85 if velocity < 15 else (0.70 if velocity < 30 else 0.55) # Aumentado de 0.60/0.50/0.35
  ```

- [ ] **Step 3: Increase Landmark (KPS) Smoothing Responsiveness**
  Adjust `kps_ema` to adapt faster to new positions.
  
  Modify `roop/ProcessMgr.py:1614`:
  ```python
  kps_ema = 0.80 if velocity < 10 else 0.90 # Aumentado de 0.65/0.80
  ```

- [ ] **Step 4: Commit changes**
  ```bash
  git add roop/ProcessMgr.py
  git commit -m "fix: improve tracking responsiveness and relax bbox clamping"
  ```

---

### Task 2: Improve Identity Fidelity and Ghost Tracking

**Files:**
- Modify: `roop/ProcessMgr.py`

- [ ] **Step 1: Adjust Hybrid Embedding Ratio**
  Change the blend ratio from 80/20 to 60/40 to allow the frame-specific embedding to contribute more to the identity, improving adaptation to different angles.
  
  Modify `roop/ProcessMgr.py:1695`:
  ```python
  hybrid_emb = (0.60 * self.master_source_embedding + 0.40 * source_face.embedding) # Cambiado de 80/20 a 60/40
  ```

- [ ] **Step 2: Increase Ghost Tracking Max Shift**
  Allow ghost tracking to follow larger movements when local tracking is briefly lost.
  
  Modify `roop/ProcessMgr.py:1343`:
  ```python
  max_shift = 80 if lost_count < 10 else 30 # Aumentado de 40/10
  ```

- [ ] **Step 3: Commit changes**
  ```bash
  git add roop/ProcessMgr.py
  git commit -m "fix: improve identity fidelity and ghost tracking range"
  ```
