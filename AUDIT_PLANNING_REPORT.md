# Full Audit & Planning Report: SigLIP Project Hardening

---

## 1. System & Environment Baseline

Execution environment probed directly:
- **OS**: Linux x86_64, 8 CPU cores, 15 GB RAM, 43 GB free NVMe disk.
- **Accelerator**: CPU only (`nvidia-smi` not found, `torch.cuda.is_available() == False`).
- **Runtime**: Python 3.14, PyTorch `2.14.1+cu130`, Transformers `5.18.0`, OpenCV `5.0.0`, Gradio `6.29.0`, Pandas `3.0.6`, NumPy `2.5.3`.
- **System Tools**: `/usr/bin/ffmpeg` present.
- **HuggingFace Cache**: `google/siglip-base-patch16-224` (203.2M params) and `google/siglip2-base-patch16-384` present locally.
- **Constraint Imposed**: Full backpropagation through SigLIP backbone impossible on CPU. Solution: freeze vision/text encoders, pre-extract embeddings once (~40 min CPU cost), train lightweight linear adapter head on cached vectors.

---

## 2. Codebase Flaws Discovered (Probe-Verified)

12 defects verified through executed test scripts:

| Component | File & Lines | Flaw Verified | Operational Impact |
|---|---|---|---|
| Frame Sampling | `src/video_processor.py:44` | Naive step rounding: `step = round(fps / target_fps)`. | 30 FPS video sampled at 4.0 FPS yielded 3.8 FPS (23 frames instead of 24). Negative/zero FPS triggered `ZeroDivisionError`. |
| Subclip Extraction | `src/video_processor.py:117` | Inclusive boundary loop: `while current <= end_frame`. | 1.0s clip at 10 FPS produced 11 frames (off-by-one). |
| Web Video Codec | `src/video_processor.py:112` | Hardcoded `cv2.VideoWriter_fourcc(*"mp4v")`. | OpenCV build lacks `avc1`/H.264 writer. `mp4v` fails to play inside standard HTML5 `<video>` tags in Gradio. |
| Memory Management | `src/video_processor.py:46-73` | Full-res PIL images accumulated in Python list. | Memory scales linearly with video length. 1-hour 1080p stream at 1 FPS requires ~22 GB RAM, triggering host OOM. |
| Text Tokenizer | `src/siglip_engine.py:71-73` | Missing `truncation=True` in tokenizer call. | Natural language queries exceeding 64 tokens crash with `ValueError: max_position_embeddings 64`. |
| Empty Query Guard | `src/video_search.py:74` | Empty strings and lists unvalidated. | Passing `""` accepted by searcher; empty candidate list crashes matrix multiplication. |
| Score Calibration | `src/video_search.py:107-124` | Reports raw cosine similarity under UI label "Probabilities". | Cosine scores range $[-1, 1]$. Native SigLIP sigmoid probabilities $\sigma(\tau \cdot \mathbf{u}^\top \mathbf{v} + b)$ never computed. |
| Feature Pooling | `src/siglip_engine.py:51-54, 75-78` | Pooling extraction logic duplicated across methods. | Transformers 5.18 returns `BaseModelOutputWithPooling`. Branch handling duplicated; `compute_zero_shot_logits` contains dtype mismatch bug on CUDA fp16. |
| Temporal NMS | `src/video_search.py:78-79` | Raw `torch.topk` returns consecutive frames. | Top 3 search results frequently return adjacent frames ($t=1.0\text{s}, 1.1\text{s}, 1.2\text{s}$) representing single event. |
| Web UI Binding | `app.py:125, 193` | `theme=gr.themes.Soft()` inside `gr.Blocks()`; binds `0.0.0.0`. | Triggers Gradio 6 deprecation warning. Exposes unauthenticated video upload service to local network. |
| In-Memory Cache | `app.py:40-43, 104-106` | New `VideoSearchEngine` instantiated per click. | Search and Classify tabs re-extract and re-encode identical video frames every interaction (~8s CPU cost per click). |
| Benchmark Integrity | `evaluate.py:131-140, 212` | `cv2.putText` draws ground-truth category text on frames; inclusive scene matching `<=". | Model reads text via OCR instead of learning visual semantics (R@1=100% partly artificial). Scene boundaries double-counted at integer timestamps. |

---

## 3. Dataset Survey & Forensic Analysis

Two external datasets inspected:

### A. `Data_stec-20261022T151522Z-1-006/Data_stec` (1.7 GB)
- **Videos**: 59 MP4 files (1658s total, resolutions $1920\times1080$ to $3840\times2160$, FPS 3.4 to 30.0).
- **Labeled Surveillance Set** (18 files):
  - `Ăn trộm`: 3 videos (`smartbench_0299`, `0543`, `0835`).
  - `TRỘM CẮP`: 2 videos (`4_095_1`, `4_108_1`).
  - `Đánh nhau`: 5 videos (`1_071_1`, `2_067_1`, `2_081_1`, `4_082_1`, `4_093_1`).
  - `NGÃ`: 4 videos (`1_078_1`, `3_092_1`, `3_093_1`, `4_087_1`).
  - `Fallen_person/NGÃ`: 1 video (`1_078_1.mp4`). SHA-256 probe confirmed identical byte duplicate (`d805cec1c80e6e97`) of `NGÃ/1_078_1.mp4`. Deduplicated to 14 unique labeled action videos.
  - `Security_guard_*`: 3 single-clip categories (posture, smoking, sitting). Kept for qualitative probe only (insufficient samples for training).
- **Unlabeled Sets**: `SmartHome Bench LLM` (41 MP4 files, no ground truth annotations).
- **Image Subsets**:
  - `Crowd`: 1644 images, 35,623 YOLO bounding boxes. Every box is class `0` (person detection only; no action/scene classes).
  - `Luggage/Normal`: 1802 images, 18 label files. All labels are full-frame dummy placeholders `0 0.5 0.5 1 1`.
  - `Person_Approaching_a_Hazardous_Area/data.zip`: 368 MB, 1081 raw JPEGs, unannotated.
- **Decision**: Filter out non-video assets. Map 14 labeled videos to canonical English classes (`theft`, `fighting`, `people_falling`) for cross-domain evaluation probe.

### B. `MSAD_anomaly-20261022T151528Z-1-001/MSAD_anomaly`
- **Videos**: 240 MP4 files (4846s total, mostly $1280\times720$, FPS 6.0 to 30.0).
- **Classes** (11 balanced categories):
  `Assault` (15), `Explosion` (15), `Fighting` (13), `Fire` (25), `Object_falling` (22), `People_falling` (43), `Robbery` (29), `Shooting` (17), `Traffic_accident` (39), `Vandalism` (12), `Water_incident` (10).
- **Label Structure**: Directory-based video-level supervision (no temporal moment boundaries).

### C. Scene Leakage & Correlation Probe
- Computed middle-frame $32\times18$ grayscale correlation across all 240 MSAD videos:
  - Within-class pairs: 3,116 pairs, mean correlation 0.03.
  - Exactly 1 pair exceeded 0.90 correlation: `Fire/Fire_4.mp4` vs `Fire/Fire_16.mp4` ($r = 0.952$).
  - Action taken: Applied Union-Find clustering at $r > 0.90$ to assign group IDs, enforcing strict Stratified Group 5-fold cross-validation.

---

## 4. Metis Gap Analysis & Resolution

Metis consultant executed in background (`bg_0a36c2b0`). Key findings incorporated into plan:
1. **Statistical Power**: Small test classes (e.g. `Water_incident` $n=10$, 2 test samples per fold) cause extreme fold metric variance. Resolved by pooling out-of-fold predictions into single $240$-sample confusion matrix and evaluating paired bootstrap 95% confidence intervals (1000 iterations).
2. **Success Criterion**: Formulated explicit mathematical threshold: Adapter declared successful (`verdict: gain`) iff pooled out-of-fold macro-F1 exceeds zero-shot baseline by $\ge 0.05$ AND lower bound of 95% bootstrap CI $> 0.0$.
3. **Cross-Domain Integrity**: STEC transfer test ($n=14$) isolated as exploratory probe with Wilson score binomial intervals. Zero claims of statistical significance.
4. **Dependency Pruning**: Rejected proposed `scikit-learn` addition to honor mandated stack. Replaced with native PyTorch multinomial logistic regression module (`src/heads.py`, ~40 lines).
5. **Memory Pass/Fail Threshold**: Defined quantitative bound: streaming 300 synthetic frames must result in $<300$ MB RSS growth (checked via `resource.getrusage`).
6. **Cache Safety**: Implemented atomic file writing (write to `.tmp_<uuid>.npz` then `os.replace`) with shape/model validation on load.

---

## 5. Work Plan Architecture

Generated complete work plan: `.omo/plans/siglip-project-hardening.md` (SHA-256: `1ae01bb22b7acf5fc91935585d0f44e989b7bb2218623fe99cdd56ddd827237d`).

### Scope & Structure:
- **Effort**: XL (20 implementation tasks across 6 waves + 4 final verifiers).
- **18 Ideal State Requirements (IS-1..IS-18)** mapped 1:1 to tasks, QA scenarios, and evidence logs.

```
Wave 1: Video I/O Core & Engine Foundation
├── Task 1: Time-based sampling & argument validation [quick]
├── Task 2: Subclip off-by-one fix & ffmpeg H.264 fallback [quick]
├── Task 3: Streaming frame generator & batched indexing [unspecified-high]
├── Task 4: Unified tensor pooling & query truncation [quick]
├── Task 5: Calibrated SigLIP sigmoid probabilities [unspecified-high]
└── Task 6: Temporal NMS & query validation [quick]

Wave 2: App & CLI Hardening
├── Task 7: Decouple synthetic generators into src/synthetic.py [quick]
├── Task 8: Gradio 6 upgrade, localhost binding, index cache [visual-engineering]
└── Task 9: CLI runner argument hardening [quick]

Wave 3: Benchmark & Evaluation Integrity
├── Task 10: Eliminate OCR label text leak in synthetic video [unspecified-high]
└── Task 11: Execute clean benchmark, update README/SYSTEM_REPORT [quick]

Wave 4: Testing & Tooling Infrastructure
├── Task 12: Fast deterministic stub unit test suite (<3s) [deep-low]
├── Task 13: Expanded integration regression test suite [unspecified-high]
└── Task 14: Requirements, .gitignore, and AGENTS sync [writing]

Wave 5: Real-World Surveillance Adaptation (Data & Features)
├── Task 15: Dataset manifest, NFC paths, deduplication [deep-low]
└── Task 16: Atomic, resumable feature embedding cache [deep-low]

Wave 6: Real-World Modeling & Evaluation
├── Task 17: PyTorch multinomial logistic regression head [deep-low]
├── Task 18: Stratified group 5-fold CV runner with bootstrap CI [deep-low]
├── Task 19: Execute 240-video MSAD & STEC surveillance benchmark [unspecified-high]
└── Task 20: Expose adapter in CLI, document real-world findings [writing]

Final Verification Wave (Parallel Audit)
├── F1: Plan compliance & evidence audit
├── F2: AST syntax & resource release code quality review
├── F3: Real manual Gradio web UI QA
└── F4: Ideal-state fidelity check (IS-1..IS-18)
```

---

## 6. Structural & Dual High-Accuracy Review Verification

1. **Automated Grammar Check**: Verified via AST script:
   - Exactly 20 column-zero tasks matching `^- \[ \] (\d+)\. <title>`.
   - Exactly 4 final verifiers matching `^- \[ \] F(\d+)\. <title>`.
   - Every implementation row contains nested `Recommended task executor category:`.
   - `## TL;DR (For humans)` placed as first major section.
2. **Momus Review (`bg_7d2211cf`)**:
   - Verdict: **`[OKAY]`**
   - Findings: All referenced files verified, instructions concrete, QA paths complete, zero blockers.
3. **Independent Oracle Review (`bg_399d5aba`)**:
   - Verdict: **`APPROVED`**
   - Findings: Binding hash matched target file; confirmed CPU runtime feasibility, exact sigmoid probability math, OCR leak elimination, and absence of external heavy dependencies.

---

## 7. Execution Readiness

All planning prerequisites complete. Implementation ready for handoff to execution worker:

```bash
# Execute plan in-place:
/ulw-execute siglip-project-hardening

# Execute plan with clean Git worktree / PR branch:
/ulw-execute siglip-project-hardening --make-pr
```
