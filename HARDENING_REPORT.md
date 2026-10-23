# Full Technical Change Report: SigLIP Project Hardening

**Branch:** `docs/audit-planning-report`  
**Git range:** `9881b88..65bb3c4` (17 commits)  
**Diff footprint:** 24 files changed, 4,285 insertions(+), 735 deletions(-)

---

## 1. Summary of 17 Commits

| Commit SHA | Type & Scope | Core Summary |
|---|---|---|
| `a180745` | `fix(video_processor)` | Time-based sampling, argument validation, half-open clip boundary, ffmpeg H.264 transcode, generator streaming. |
| `77bd982` | `fix(siglip_engine)` | Centralized `_pool_features`, 64-token truncation, calibrated `score_logits()`. |
| `9e2ac07` | `feat(video_search)` | Batched streaming index, sigmoid probabilities with templating, greedy temporal NMS. |
| `aa62962` | `refactor(synthetic)` | Decoupled dummy and multi-scene benchmark video generators into `src/synthetic.py`. |
| `43780a9` | `feat(dataset)` | 240 MSAD + 14 STEC manifest loader, NFC Unicode path handling, SHA-256 dedupe, background correlation camera clustering. |
| `dd5143a` | `feat(heads)` | Native PyTorch multinomial logistic regression adapter (`LinearAdapterHead`), prompt ensembling, atomic checkpointing. |
| `2870bd5` | `feat(app)` | Gradio 6 modernization, localhost binding (127.0.0.1:7860), in-memory thread-safe LRU index cache, isolated clip cleanup. |
| `31c76cb` | `fix(main)` | CLI `--min-gap`, explicit `--dummy-out`, robust error handling without traceback leaks. |
| `f5cf5bf` | `fix(evaluate)` | Removed OCR text overlays (`clean_no_overlay`), half-open `[start, end)` evaluation interval, separated timing. |
| `2a079f0` | `feat(features)` | Atomic resumable `.npz` feature cache (`cache/features/`) with format/model validation. |
| `f12efb0` | `test(unit)` | 18 fast deterministic unit tests (< 5s) with `StubSigLIPEngine`, flat memory assertions (< 300 MB). |
| `1a804bc` | `docs(evaluate)` | Regenerated honest benchmark metrics, synchronized `README.md` and `SYSTEM_REPORT.md`. |
| `0770b17` | `test(pipeline)` | Ephemeral `tempfile.TemporaryDirectory` fixtures, color segment ordering assertions, probability bounds checks. |
| `fd6c62d` | `chore(config)` | Pinned dependencies in `requirements.txt`, updated `.gitignore`, documented two-tier testing convention in `AGENTS.md`. |
| `d97f77b` | `feat(evaluate_real)` | 5-fold cross-validation evaluation runner with paired bootstrap CI and Wilson score intervals. |
| `30794cd` | `docs(surveillance)` | Exposed `--adapter-checkpoint` in `main.py`, added real surveillance evaluation sections to docs. |
| `65bb3c4` | `refactor(typing)` | Completed explicit return type annotations across engine, app, and unit tests. |

---

## 2. File-by-File Changes

### Component 1: Video I/O Core

#### `src/video_processor.py` (+251, -70)
- **Time-based frame sampling**: Replaced `step = round(fps / target_fps)` integer rounding with timestamp progression (`next_timestamp += 1.0 / target_fps`).
- **Skipped frame decode optimization**: Used `cap.grab()` on non-target frames instead of full BGR `cap.read()`.
- **Strict argument validation**:
  - `target_fps`: Must be finite and positive (`math.isfinite(target_fps) and target_fps > 0`). Rejects `0`, negative, `NaN`, `+inf`, `-inf` with `ValueError`.
  - `max_frames`: Must be `None` or integer `> 0`.
- **Subclip boundary fix**: Frame extraction loop converted from inclusive `while current <= end_frame` to half-open `while current < end_frame`. Interval `dt` produces exactly `round(dt * fps)` frames.
- **H.264 browser playback transcode**:
  - Raw clip written with `cv2.VideoWriter_fourcc(*"mp4v")`.
  - If `ffmpeg` detected on system `PATH` via `shutil.which('ffmpeg')`, transcodes atomically to H.264 `yuv420p` + `+faststart` into `.tmp_<uuid>.mp4` before `os.replace`.
  - Falls back to `mp4v` if ffmpeg absent or fails.
- **Streaming frame generator**:
  - Added `iter_frames(video_path: str) -> Iterator[Dict[str, Union[Image.Image, float, int]]]`.
  - `extract_frames()` wrapped as `list(self.iter_frames(video_path))`.
- **Resource safety**: All `cv2.VideoCapture` and `cv2.VideoWriter` instances wrapped in `try...finally: cap.release()` / `out.release()`.

---

### Component 2: Engine & Moment Retrieval

#### `src/siglip_engine.py` (+69, -27)
- **Unified feature pooling**: Extracted private static helper `_pool_features(features) -> torch.Tensor` handling `BaseModelOutputWithPooling` (transformers 5.x), index `[0]`, and raw tensors. Shared by `encode_images` and `encode_texts`.
- **Tokenizer truncation**: Added `truncation=True, max_length=64` to `self.processor(text=texts, ...)` to prevent index errors on queries > 64 tokens.
- **Empty input handling**: Empty text list returns `torch.empty((0, hidden_size))`. Empty image list returns `torch.empty((0, hidden_size))`.
- **Calibrated sigmoid probabilities**:
  - Implemented `score_logits(image_embeds, text_embeds) -> Tuple[torch.Tensor, torch.Tensor]`.
  - Cosine similarity: `sims = image_embeds @ text_embeds.T`.
  - Sigmoid probability: `probs = torch.sigmoid(self.model.logit_scale.exp() * sims + self.model.logit_bias)`.
  - Removed unused and buggy `compute_zero_shot_logits`.

#### `src/video_search.py` (+140, -32)
- **Streaming batched index**: `index_video(video_path, batch_size=16)` consumes `iter_frames()` in batches of 16, encodes via `engine.encode_images`, and accumulates only embeddings and metadata. Flat memory footprint.
- **Zero-shot templating**: Added `template: Optional[str] = "This is a photo of {}."`. Formats labels lacking `{}` before text encoding. Returns `mean_prob` and `max_prob` alongside cosine similarities.
- **Greedy temporal NMS**: Added `min_gap_sec: float = 2.0` to `search_moments()`. Suppresses adjacent frames within `min_gap_sec` from the same video. `min_gap_sec=0.0` returns plain top-k.
- **Query validation**: Strips query; empty or whitespace string raises `ValueError`. Rejects non-positive `top_k` and negative `min_gap_sec`.

---

### Component 3: Web UI, CLI & Synthetic Data

#### `src/synthetic.py` (New file, 200 lines)
- Decoupled synthetic generators from CLI/evaluation scripts.
- `generate_dummy_video(output_path, duration_sec)`: Directory creation handles bare filenames via `os.path.dirname(os.path.abspath(output_path))`.
- `generate_benchmark_video(video_path, duration_per_scene, fps, with_text_overlay=False)`:
  - Generates 8 multi-scene visual patterns.
  - Category label text overlay disabled by default (`with_text_overlay=False`) to prevent OCR leakage.

#### `app.py` (+308, -118)
- **Gradio 6 modernization**: Exposes `build_app() -> gr.Blocks`. Moved `theme` to `launch(theme=gr.themes.Soft(), server_name=..., server_port=...)`.
- **Localhost binding**: Defaults to `127.0.0.1:7860` via `os.getenv('SIGLIP_APP_HOST', '127.0.0.1')`.
- **Thread-safe LRU index cache**: Caches up to 4 `VideoSearchEngine` instances keyed by `(abs_path, mtime_ns, size, fps, model_name)`. Sequential search and classify queries reuse the same index.
- **Temp folder isolation**: Subclips saved under `/tmp/siglip_clip_<uuid>/`. Automatic pruning for folders older than 10 minutes.
- **UI scale guard**: Rejects videos yielding > 3600 frames on CPU.
- **Calibrated UI table**: Displays `Mean probability %` and `Max probability %` from native SigLIP sigmoid probabilities.

#### `main.py` (+236, -78)
- Added `--min-gap` argument for temporal NMS in CLI search.
- Added `--adapter-checkpoint` argument to run inference using trained PyTorch adapter heads.
- Safe `--video` handling: Non-existent paths exit code 2 with clear error message; never regenerates or clobbers `sample_data/demo.mp4`.
- Added `--dummy-out` to control synthetic video destination.

---

### Component 4: Real Surveillance Pipeline & Adaptation

#### `src/dataset.py` (New file, 300 lines)
- `VideoSample` dataclass container.
- `LABEL_MAP`: Maps Vietnamese folder names (NFC-normalized) to canonical English classes (`Ăn trộm` / `TRỘM CẮP` -> `theft`, `Đánh nhau` -> `fighting`, `NGÃ` / `Fallen_person` -> `people_falling`).
- `load_surveillance_manifest(msad_root, stec_root)`: Scans strictly `*.mp4` files. Performs SHA-256 deduplication (deduped `1_078_1.mp4`).
- Camera clustering: Extracted middle frames (32x18 grayscale), computed Pearson correlation, and clustered pairs with `corr > 0.90` via Union-Find (groups `Fire_4` and `Fire_16`; 239 total clusters across 240 MSAD clips).
- `stratified_group_kfold`: Numpy-only 5-fold cross-validation partitioner ensuring no camera group straddles train and test folds.

#### `src/features.py` (New file, 258 lines)
- `extract_or_load_features(video_path, engine, processor, cache_dir)`:
  - Cache format: `<cache_dir>/<model_sanitized>/<sha256>_fps<fps>_max<max>.npz`.
  - Atomic writes: Saves to `.tmp_<uuid>.npz` before `os.replace`.
  - Validation on load: Validates dimensions, finiteness, unit norm, model name, and sampling rate. Automatically removes corrupted files and regenerates.
- `extract_many()`: Resumable batch extraction runner.

#### `src/heads.py` (New file, 287 lines)
- `LinearAdapterHead`: Single PyTorch linear projection (`768 -> num_classes`) with bias. Dimension check asserts 2D input.
- `train_head()`: Full-batch L-BFGS optimizer with strong-wolfe line search, L2 weight decay, CrossEntropyLoss, optional class weights, and early stopping on validation loss.
- `encode_prompt_ensemble()`: Multi-template prompt ensembling across 4 canonical surveillance templates with L2 normalization.
- `pool_video_embedding()`: Supports `mean` and `topk_mean` pooling with L2 normalization.
- `save_head()` and `load_head()`: Atomic checkpoint serialization refusing mismatched model IDs or class lists.

#### `src/metrics.py` (New file, 103 lines)
- `wilson_score_interval(k, n, confidence=0.95)`: Wilson binomial score confidence interval.
- `paired_bootstrap_test(y_true, y_adapter, y_zero_shot, num_classes, n_resamples=1000)`: 95% paired bootstrap confidence interval for Macro-F1 delta.
- `evaluate_stec_probe()`: Cross-domain transfer probe with sample size disclaimers ($n < 10$).

#### `evaluate_real.py` (New file, 231 lines)
- Headless 5-fold cross-validation benchmark runner for real surveillance data.
- Outputs verified JSON schema to `eval_real_results.json`.
- Saves final full-dataset adapter checkpoint to `checkpoints/adapter_msad_base224.pt`.

---

### Component 5: Benchmarks & Documentation Sync

#### `evaluate.py` (+655, -284)
- Removed local synthetic video generation; imports `src.synthetic.generate_benchmark_video`.
- Added `--with-text-overlay` flag for OCR leakage baseline control.
- Converted all scene boundary matching from inclusive `<=` to half-open `[start, end)`.
- Separated frame extraction latency from model inference latency.
- Atomic output writing via temporary files.

#### `eval_results.json` (+330, -70)
- Updated with uncorrupted synthetic benchmark numbers without text overlay (`clean_no_overlay`):
  - Recall@1: 87.50%
  - Recall@3: 91.67%
  - Mean Reciprocal Rank (MRR): 0.8958
  - Mean Absolute Error (MAE to interval): 1.12s
  - Average Query Latency: 477.54 ms (CPU)

#### `eval_real_results.json` (New artifact, 3.2 KB, gitignored)
- Verified real-world surveillance benchmark:
  - MSAD 11-Class Zero-Shot Macro-F1: 12.53%
  - MSAD 11-Class Prompt-Ensemble Macro-F1: 23.38%
  - MSAD 11-Class 5-Fold Linear Adapter Macro-F1: 19.99%
  - Adapter improvement over Zero-Shot: +7.46%
  - Paired Bootstrap 95% CI: `[0.0242, 0.1232]` (lower bound > 0)
  - Formal statistical verdict: `gain`
  - STEC cross-domain exploratory probe: 14 clips, small sample disclaimer ($n < 10$).

#### `README.md` & `SYSTEM_REPORT.md`
- Synchronized all benchmark tables with measured numbers.
- Added historical note explaining prior 100% R@1 was an artifact of `cv2.putText` OCR leakage.
- Added Vietnamese section: *"Đánh giá trên dữ liệu Camera Giám sát Thực tế (Real-World Surveillance Evaluation)"*.
- Added hardware timing breakdown and CLI adapter usage examples.
- Privacy compliance: Zero real surveillance video frames, faces, or identifying screenshots added.

---

### Component 6: Testing, Tooling & Configuration

#### `tests/test_unit.py` (New file, 581 lines)
- 18 fast deterministic unit tests (< 5s) running offline with `StubSigLIPEngine`:
  - Frame sampling frequency across arbitrary native/target FPS.
  - VideoProcessor validation (NaN, inf, <= 0, max_frames).
  - Half-open subclip duration and boundary clamping.
  - Greedy temporal NMS and query truncation.
  - Streaming batched indexing and append behavior.
  - Zero-shot classification contract and prompt templating.
  - Memory bounds test: 300 frames (640x360) showed `0.00 MB` peak RSS growth.
  - Dataset grouping, NFC Unicode normalization, and StratifiedGroupKFold partitioning.
  - Synthetic video bare filename generation in temp directory.

#### `tests/test_pipeline.py` (+108, -27)
- Real-model integration tests (`HF_HUB_OFFLINE=1`).
- Ephemeral fixtures using `tempfile.TemporaryDirectory`. Zero leftover test files.
- Color segment ordering assertions (frames 0-29 blue, frames 30-59 red).
- Sigmoid probability bounds validation in `[0.0, 1.0]`.

#### `tests/test_heads.py` (New file, 233 lines)
- 14 tests for `LinearAdapterHead`, L-BFGS training convergence, early stopping, prompt ensembling, pooling modes, and atomic serialization.

#### `tests/test_evaluate_real.py` (New file, 193 lines)
- 5 tests for CLI argument parsing, missing root error handling, mock cross-validation pipeline, Wilson intervals, and paired bootstrap resampling.

#### `requirements.txt` & `.gitignore`
- `requirements.txt`: Updated pins (`gradio>=6.0.0`, `torch>=2.0.0`, `transformers>=4.40.0`, `opencv-python>=4.8.0`, `pillow>=9.0.0`, `numpy>=1.24.0`, `pandas>=2.0.0`). Zero `scikit-learn` dependency.
- `.gitignore`: Ignores `cache/`, `checkpoints/`, `*.npz`, `tests/*.mp4`, `eval_real_results.json`, and local dataset folder patterns.

#### `AGENTS.md`, `src/AGENTS.md`, `tests/AGENTS.md`
- Documented two-tier testing convention:
  - Fast Unit Lane (`tests/test_unit.py`): Deterministic stubs, < 5s, zero model downloads.
  - Real-Model Integration Lane (`tests/test_pipeline.py`): Real weights, offline (`HF_HUB_OFFLINE=1`).
- Reconciled "no mocks" rule: Applies strictly to integration testing tier; unit testing tier uses `StubSigLIPEngine`.
- Documented new module architecture, conventions, and anti-patterns.

---

## 3. Specifics of the Latest Commit (`65bb3c4`)

Commit `65bb3c4` (`refactor(typing): complete return type annotations across engines, app, and unit tests`) closed the final code-quality audit item from Wave F2:
- Added explicit return type annotation `-> Tuple[str, List[Any], Optional[pd.DataFrame], Optional[str]]` to `app.process_video_and_search`.
- Added explicit return type annotation `-> Tuple[str, Optional[pd.DataFrame]]` to `app.classify_video`.
- Added `-> None` annotations to constructors across `SigLIPEngine`, `VideoProcessor`, and `VideoSearchEngine`.
- Added typing annotations to helper functions and fixture setup methods in `tests/test_unit.py`.

---

## 4. Final Verification State

- **Unit & Integration Test Suite**: 39/39 tests pass in 4.87s (`python -m unittest discover tests`).
- **All 20 Implementation Tasks + F1-F4**: Marked complete `[x]` in `.omo/plans/siglip-project-hardening.md`.
- **Evidence Files**: 40 verification evidence files present in `.omo/evidence/siglip-project-hardening/`.
- **Working Tree**: Completely clean; git status tracked cleanly with origin.
