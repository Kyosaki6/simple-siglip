# PROJECT KNOWLEDGE BASE

**Generated:** 2026-10-22
**Commit:** 826d08b
**Branch:** main

## OVERVIEW
SigLIP video search and zero-shot action/scene classification pipeline. Core stack: PyTorch, HuggingFace Transformers (`google/siglip-base-patch16-224`), OpenCV, PIL, Gradio.

## STRUCTURE
```
.
├── src/
│   ├── siglip_engine.py    # SigLIP model wrapper for vision/text embeddings
│   ├── video_processor.py  # OpenCV video sampling, seeking, and clip exporter
│   ├── video_search.py     # In-memory frame vector index and moment ranker
│   ├── features.py         # Resumable atomic embedding cache (.npz)
│   ├── heads.py            # PyTorch linear adapter head & prompt ensembling
│   ├── dataset.py          # Surveillance video manifest, NFC paths, grouping
│   └── synthetic.py        # Deterministic benchmark & dummy video generators
├── tests/
│   ├── test_unit.py        # Fast unit tests (< 10s, deterministic stubs)
│   ├── test_pipeline.py    # Real-model integration tests (HF_HUB_OFFLINE=1)
│   └── test_heads.py       # Adapter head convergence & dimension tests
├── sample_data/            # Benchmark video clips
├── app.py                  # Gradio dual-tab UI (search + zero-shot classify)
├── evaluate.py             # Recall@K, MRR, MAE benchmark evaluator
├── evaluate_real.py        # 5-fold CV evaluation with bootstrap CI
└── main.py                 # CLI runner for query search, classify, synthetic video
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Web UI & Gradio endpoints | `app.py` | Cached engine via `get_engine()`, temp clip generator |
| CLI execution & synthetic video | `main.py` | `generate_dummy_video()`, argument parser |
| Benchmark metrics & ablation | `evaluate.py` | `evaluate_model()`, synthetic scene ground-truth |
| Model inference & embeddings | `src/siglip_engine.py` | `SigLIPEngine`: image/text L2 normalized vectors |
| Video decoding & clipping | `src/video_processor.py` | `VideoProcessor`: FPS step sampling, timestamp frame seek |
| Moment retrieval & indexing | `src/video_search.py` | `VideoSearchEngine`: dot-product similarity, top-k |
| Embedding cache (.npz) | `src/features.py` | `extract_or_load_features()`: atomic write, corruption check |
| Adapter head & ensembling | `src/heads.py` | `LinearAdapterHead`: PyTorch L-BFGS/Adam classifier |
| Surveillance dataset & splits | `src/dataset.py` | `load_surveillance_manifest()`, `LABEL_MAP`, group k-fold |
| Synthetic video generation | `src/synthetic.py` | `generate_dummy_video()`, `generate_benchmark_video()` |

## CODE MAP
| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `SigLIPEngine` | Class | `src/siglip_engine.py:8` | 4 files | Model loader & L2 normalized encoder |
| `VideoProcessor` | Class | `src/video_processor.py:8` | 4 files | OpenCV video sampler & snippet exporter |
| `SearchResult` | Dataclass | `src/video_search.py:13` | 2 files | Timestamped match struct (ts, score, idx) |
| `VideoSearchEngine` | Class | `src/video_search.py:20` | 4 files | In-memory frame vector index & ranker |
| `extract_or_load_features` | Function | `src/features.py` | 3 files | Atomic feature caching with SHA-256 keys |
| `LinearAdapterHead` | Class | `src/heads.py` | 2 files | Multinomial logistic regression adapter |
| `load_surveillance_manifest` | Function | `src/dataset.py` | 2 files | Manifest loader with NFC normalization & deduplication |
| `generate_dummy_video` | Function | `src/synthetic.py` | 3 files | Synthetic RGB 3-scene test video generator |
| `get_engine` | Function | `app.py:18` | `app.py` | Module-level cached engine factory |
| `evaluate_model` | Function | `evaluate.py:149` | `evaluate.py` | Full R@K, MRR, latency benchmark loop |

## TESTING CONVENTIONS (TWO-TIER)
- **Fast Unit Lane (`tests/test_unit.py`)**:
  - Deterministic stubs (`StubSigLIPEngine`) generating normalized pseudo-embeddings.
  - Zero model downloads (`AutoModel.from_pretrained` is explicitly forbidden).
  - Fast turnaround (< 10s execution for unit checks).
  - Covers edge cases, memory bounds, stream processing, cache corruption, and manifest grouping.
- **Integration Lane (`tests/test_pipeline.py`)**:
  - Uses real `google/siglip-base-patch16-224` model weights with `HF_HUB_OFFLINE=1`.
  - Verifies PyTorch dtype, tensor shapes, and live search pipeline ranking.
- **Reconciliation with "No Mocks" Rule**:
  - The "no mocks" rule applies strictly to the integration tier (`tests/test_pipeline.py`) to guarantee real-weight model inference and tensor behavior.
  - The fast unit tier (`tests/test_unit.py`) uses lightweight deterministic stubs (`StubSigLIPEngine`) so unit suites execute offline, instantly, and without GPU or network access.

## CONVENTIONS
- Mandatory typing: `Optional`, `List`, `Dict`, `Tuple` on all signatures.
- Cosine similarity via L2-normalization (`F.normalize(p=2, dim=-1)`) followed by matrix multiply `@`.
- Dependency injection pattern: engines/processors pass optional instances with sensible defaults.
- Floating timestamps rounded to 2 decimal places.

## ANTI-PATTERNS (THIS PROJECT)
- DO NOT duplicate tensor pooling logic: handle `pooler_output` vs raw tensor uniformly.
- DO NOT hardcode magic FPS: use parameterization over raw `25.0` / `1.0` defaults.
- NEVER hold video capture unreleased: close `cv2.VideoCapture` / `cv2.VideoWriter`.
- NEVER bypass L2 normalization before vector dot product.
- DO NOT invoke `AutoModel.from_pretrained` in `tests/test_unit.py`.

## COMMANDS
```bash
# Setup
pip install -r requirements.txt

# Web Interface
python3 app.py

# CLI Search & Classify
python3 main.py --video sample_data/demo.mp4 --query "blue ocean with water"
python3 main.py --generate-dummy

# Two-Tier Tests
# Fast Unit Lane (< 10s, no model download)
python3 -m unittest tests/test_unit.py
# Real-Model Integration Lane (real weights, offline)
export HF_HUB_OFFLINE=1 PYTHONPATH=.
python3 -m unittest tests/test_pipeline.py

# Evaluation Benchmark
python3 evaluate.py
```

## NOTES
- HuggingFace model `google/siglip-base-patch16-224` defaults to CPU float32 or CUDA float16.
- In-memory index: `VideoSearchEngine.frame_embeddings` stored as single PyTorch tensor.
