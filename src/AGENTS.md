# SRC MODULE KNOWLEDGE BASE

## OVERVIEW
Core multimodal retrieval and classification pipeline: SigLIP model wrapper, OpenCV frame sampler, in-memory cosine vector indexer, atomic feature cache, PyTorch linear adapter heads, surveillance dataset parser, and synthetic video generators.

## WHERE TO LOOK
| Component | Location | Role |
|-----------|----------|------|
| `SigLIPEngine` | `src/siglip_engine.py:8` | HuggingFace SigLIP model wrapper for vision/text L2 embedding extraction |
| `VideoProcessor` | `src/video_processor.py:8` | OpenCV frame decoding at target FPS, timestamp seeking, subclip writer |
| `SearchResult` | `src/video_search.py:12` | Dataclass container for retrieved moment (timestamp, frame_idx, score, video_path) |
| `VideoSearchEngine` | `src/video_search.py:20` | Orchestrates frame extraction, embedding accumulation, dot-product similarity search |
| `extract_or_load_features` | `src/features.py:72` | Atomic `.npz` caching under `cache/features/` with SHA-256 keys and corruption detection |
| `LinearAdapterHead` | `src/heads.py:22` | PyTorch linear adapter layer (`in_features=768, out_features=num_classes`) with L-BFGS, optional class weights and text-embedding init |
| `encode_prompt_ensemble` | `src/heads.py:100` | Multi-template prompt ensembling across canonical surveillance templates with L2 norm |
| `load_surveillance_manifest`| `src/dataset.py:140` | Scans MSAD & STEC `.mp4` files, NFC path normalization, SHA-256 deduplication |
| `stratified_group_kfold` | `src/dataset.py:230` | Group-aware stratified cross-validation splitter avoiding camera angle data leakage |
| `generate_benchmark_video`| `src/synthetic.py:50` | Synthetic video generator creating multi-scene visual patterns for reproducible testing |

## PIPELINE ARCHITECTURE & DATA FLOW
1. **Video Ingestion & Sampling**:
   - `VideoProcessor.extract_frames(video_path)` steps through video at `target_fps = 1.0` (step = `round(fps / target_fps)`), converts BGR to RGB PIL images, tracks float `timestamp`.
   - Streaming generator mode (`stream_frames`) avoids unbounded memory growth on long clips.
2. **Feature Extraction & Atomic Caching (`src/features.py`)**:
   - Computes video SHA-256 hash and resolves cache path: `cache/features/<model_sanitized>/<sha256>_fps{fps}_max{max}.npz`.
   - On cache miss: extracts frames, runs `SigLIPEngine.encode_images`, writes atomically to `.tmp_<uuid>.npz`, and renames via `os.replace`.
   - On cache hit: verifies feature dimensions and model identifier; auto-regenerates if corrupt.
3. **Vision & Text Encoding (`src/siglip_engine.py`)**:
   - Vision: feeds batches to `google/siglip-base-patch16-224`, normalizes with L2 (`F.normalize(p=2, dim=-1)`), returns CPU tensor `(N, 768)`.
   - Text: encodes text queries with padding, L2-normalizes to `(M, 768)`.
4. **Adapter Heads & Prompt Ensembling (`src/heads.py`)**:
   - `LinearAdapterHead`: maps 768-dim pooled video embeddings to action/anomaly class logits.
   - `encode_prompt_ensemble`: averages normalized text embeddings across 4 templates ("a surveillance video of {}", etc.) and re-normalizes.
5. **Surveillance Dataset & Grouping (`src/dataset.py`)**:
   - Parses MSAD (11 classes) and STEC (Vietnamese labels mapped via NFC).
   - Groups clips from identical camera viewpoints via perceptual background correlation (> 0.90) to prevent cross-fold test leakage.

## CONVENTIONS
- Return PyTorch CPU tensors from engine methods (`encode_images`, `encode_texts`) for downstream storage.
- All embedding tensors MUST be L2-normalized (`F.normalize(p=2, dim=-1)`) immediately after model forward.
- Keep `VideoProcessor` utility operations (`extract_frame_at_timestamp`, `clip_video`) as `@staticmethod`.
- Use lazy dependency injection: `self.engine = engine or SigLIPEngine()`.
- Feature cache writes MUST be atomic using temporary UUID files and `os.replace`.
- All string paths and Vietnamese folder names MUST be NFC normalized (`unicodedata.normalize('NFC', ...)`).
- Two-Tier Testing: Code in `src/` must be testable via deterministic stubs (`StubSigLIPEngine`) in `tests/test_unit.py` without model downloads (< 10s execution). Real model integration tests in `tests/test_pipeline.py` run in offline mode (`HF_HUB_OFFLINE=1`). "No mocks" rule applies strictly to the integration lane; unit lane requires stubs for hermetic validation.

## ANTI-PATTERNS (THIS SUBDIR)
- DO NOT pass raw unnormalized embeddings into `frame_embeddings` or similarity computation.
- DO NOT return BGR frames: always convert `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before PIL wrapping.
- NEVER forget `cap.release()` or `out.release()` in OpenCV operations.
- AVOID running model inference outside `torch.no_grad()` context.
- DO NOT hardcode device: default to CUDA when available, fallback to CPU.
- NEVER write partially completed `.npz` or checkpoint files directly to final destination.
- DO NOT import `scikit-learn` in `heads.py` or `dataset.py`; use native PyTorch and NumPy routines.

## COMMANDS
```bash
# Fast Unit Lane (< 10s, no model download)
python3 -m unittest tests/test_unit.py

# Real-Model Integration Lane (real weights, offline)
export HF_HUB_OFFLINE=1 PYTHONPATH=.
python3 -m unittest tests/test_pipeline.py
```
