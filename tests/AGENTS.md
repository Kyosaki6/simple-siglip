# TESTS MODULE KNOWLEDGE BASE

## OVERVIEW
Two-tier test suite: fast hermetic unit tests with deterministic stubs (`tests/test_unit.py`), focused adapter head tests (`tests/test_heads.py`), and real-model pipeline integration tests (`tests/test_pipeline.py`).

## WHERE TO LOOK
| Test Target | Location | Scope |
|-------------|----------|-------|
| Fast Unit Suite | `tests/test_unit.py` | Hermetic unit tests (< 10s): stubs, streaming memory limits, edge cases, manifest parsing, grouping |
| Real-Model Integration | `tests/test_pipeline.py` | End-to-end integration: real SigLIP weights (`HF_HUB_OFFLINE=1`), video decoding, live moment search |
| Adapter Head Tests | `tests/test_heads.py` | Training convergence, loss decreases, dimension guards, and checkpoint serialization |

## TWO-TIER TESTING ARCHITECTURE
1. **Tier 1: Fast Unit Tests (`tests/test_unit.py`)**:
   - **Deterministic Stubs**: Uses `StubSigLIPEngine` producing deterministic L2-normalized pseudo-embeddings derived from image RGB colors or text hashes.
   - **Zero Model Downloads**: `transformers.AutoModel.from_pretrained` is monkeypatched to raise an `AssertionError` if invoked.
   - **Speed Budget**: Executes in < 10 seconds without GPU or external network access.
   - **Coverage**: Validates frame extraction, timestamp rounding, cache hit/miss/corruption, streaming memory ceilings (bounded RSS), manifest parsing with NFC normalization, and camera grouping.
2. **Tier 2: Real-Model Integration Tests (`tests/test_pipeline.py`)**:
   - Runs against real `google/siglip-base-patch16-224` model weights on CPU with `export HF_HUB_OFFLINE=1 PYTHONPATH=.`.
   - Validates genuine HuggingFace weight loading, tensor types, output dimensions, and moment retrieval accuracy.

## RECONCILING THE "NO MOCKS" RULE
- **Historical rule**: "DO NOT mock `SigLIPEngine`."
- **Scope clarification**:
  - In **`tests/test_pipeline.py`**, mocking `SigLIPEngine` remains strictly **FORBIDDEN**. The integration suite's primary purpose is to act as a smoke test for actual model weights, real tokenizer behavior, and true PyTorch execution.
  - In **`tests/test_unit.py`**, deterministic stubs (`StubSigLIPEngine`) are **MANDATORY**. Calling the actual 200M+ parameter model during unit testing wastes time, requires internet/cache presence, and slows iteration. The stub provides mathematical invariants (normalized 768-dim float32 vectors) while isolating pipeline logic from model weights.

## FIXTURE DESIGN & TEST DATA
- **Synthetic Video Fixture**: `setUpClass` in `test_pipeline.py` generates `tests/test_dummy.mp4` via OpenCV `cv2.VideoWriter`:
  - 10 FPS, 20 total frames (2.0s duration), 160x120 resolution.
  - Distinct colored halves: frames 0-9 blue RGB `(255, 0, 0)`, frames 10-19 red RGB `(0, 0, 255)`.
  - Gives deterministic ground truth for retrieval tests ("blue picture" vs "red picture").
- **Automatic Teardown**: `tearDownClass` unlinks `tests/test_dummy.mp4`. Method-level subclips (`tests/test_clip.mp4`) are removed immediately after assertion.
- **Gitignore Compliance**: All test clips match `tests/*.mp4` and are excluded from git commits.

## CONVENTIONS
- Standard `unittest.TestCase` class structure.
- Always run unit suite first for rapid local validation before running the full integration pipeline.
- Keep fixtures ephemeral: create in temp directories or unlink in `tearDownClass` / `tearDown`.

## ANTI-PATTERNS (THIS SUBDIR)
- NEVER leave generated test video files on disk after test suite execution.
- DO NOT call `AutoModel.from_pretrained` inside `tests/test_unit.py`.
- DO NOT mock `SigLIPEngine` in `tests/test_pipeline.py` (real model verification required).
- AVOID adding external network calls; keep dependencies strictly local or synthetic.

## COMMANDS
```bash
# Fast Unit Lane (< 10s, no model download)
python3 -m unittest tests/test_unit.py

# Real-Model Integration Lane (real weights, offline)
export HF_HUB_OFFLINE=1 PYTHONPATH=.
python3 -m unittest tests/test_pipeline.py

# Adapter Head Tests
python3 -m unittest tests/test_heads.py

# Run all test discovery
python3 -m unittest discover tests
```
