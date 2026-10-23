# Technical Architecture & System Report: Demo Siglip

## 1. Executive Summary

`Demo Siglip` implements a video understanding pipeline using Google's SigLIP (Sigmoid Loss for Language Image Pre-training) backbone (`google/siglip-base-patch16-224` / `google/siglip-so400m-patch14-384`).

Two primary capabilities:
1. **Text-to-Moment Retrieval**: Identifies timestamps matching arbitrary natural language descriptions without pre-defined labels.
2. **Zero-Shot Video Classification**: Evaluates multi-class semantic categories over whole video via temporal frame aggregation without fine-tuning.

---

## 2. Directory Structure & File Map

```
Demo Siglip/
├── sample_data/
│   └── demo.mp4               # Synthetic multi-scene validation video
├── src/
│   ├── siglip_engine.py       # Model wrapper (feature extraction, normalization, device handling)
│   ├── video_processor.py     # Frame sampling, time-offset mapping, subclip writer
│   └── video_search.py        # Matrix indexing, vector dot product, top-k retrieval
├── tests/
│   └── test_pipeline.py       # Automated unit tests for all components
├── evaluate.py                # Full synthetic benchmark evaluation runner
├── eval_results.json          # Machine-readable evaluation benchmark metrics
├── app.py                     # Gradio web user interface with video preview
├── main.py                    # CLI runner + synthetic video generator
├── README.md                  # Quickstart documentation
└── requirements.txt           # Python dependencies
```

---

## 3. Theoretical Foundations: SigLIP vs. CLIP

Traditional CLIP applies cross-entropy loss over softmax-normalized similarity matrices across batch size $B$:

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2B} \sum_{i=1}^B \left( \log \frac{e^{\tau \mathbf{u}_i^\top \mathbf{v}_i}}{\sum_{j=1}^B e^{\tau \mathbf{u}_i^\top \mathbf{v}_j}} + \log \frac{e^{\tau \mathbf{u}_i^\top \mathbf{v}_i}}{\sum_{j=1}^B e^{\tau \mathbf{u}_j^\top \mathbf{v}_i}} \right)$$

SigLIP replaces global softmax normalization with pairwise binary cross-entropy (Sigmoid loss):

$$\mathcal{L}_{\text{SigLIP}} = -\sum_{i=1}^{B} \sum_{j=1}^{B} \log \sigma \left( z_{ij} \cdot (\tau \mathbf{u}_i^\top \mathbf{v}_j + b) \right)$$

where $z_{ij} = 1$ if $i = j$ else $-1$, with learned temperature $\tau$ and learned bias $b$.

### Engineering Implications in this Project
- **Memory efficiency**: No batch-wide normalization requirement.
- **Inference consistency**: Pairwise image-text matching probabilities stay decoupled from batch compositions.

---

## 4. Module-by-Module Technical Breakdown

### 4.1. `src/video_processor.py` (`VideoProcessor`)

Handles all video I/O, frame sampling, temporal calculations, and file export via OpenCV and Pillow.

#### Key Methods:

| Method | Inputs | Outputs | Description |
|---|---|---|---|
| `__init__` | `target_fps=1.0`, `max_frames=None` | `None` | Configures sampling density and extraction ceiling. |
| `extract_frames` | `video_path: str` | `List[Dict]` | Reads video via `cv2.VideoCapture`. Computes frame stride: $\text{step} = \max(1, \lfloor \text{native\_fps} / \text{target\_fps} \rceil)$. Pulls frames, maps to RGB, formats as PIL Image, computes float timestamp: $t = \frac{\text{frame\_idx}}{\text{native\_fps}}$. |
| `extract_frame_at_timestamp` | `video_path: str`, `timestamp_sec: float` | `PIL.Image \| None` | Direct frame seek using `cv2.CAP_PROP_POS_FRAMES`. Avoids full video re-read. |
| `clip_video` | `video_path, start_sec, end_sec, output_path` | `str` | Slices segment $[t_{\text{start}}, t_{\text{end}}]$ into standalone `.mp4` using `cv2.VideoWriter` with `mp4v` codec. |

---

### 4.2. `src/siglip_engine.py` (`SigLIPEngine`)

Encapsulates Hugging Face `AutoProcessor` and `AutoModel` for Google SigLIP.

#### Key Mechanics:
- **Device & Precision Allocation**:
  - `CUDA` available $\rightarrow$ `torch.float16` on `cuda`.
  - CPU only $\rightarrow$ `torch.float32` on `cpu`.
  - Model set strictly to `eval()` mode. Inference runs under `@torch.no_grad()`.
- **`encode_images(images, batch_size=16)`**:
  - Batches PIL images into `AutoProcessor(images=batch, return_tensors="pt")`.
  - Calls `model.get_image_features()`.
  - Enforces $L_2$ normalization:
    $$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
  - Concatenates mini-batches onto CPU to avoid GPU VRAM exhaustion.
- **`encode_texts(texts)`**:
  - Passes string list into `AutoProcessor(text=texts, padding="max_length", return_tensors="pt")`.
  - Extracts text embeddings with `model.get_text_features()`.
  - Applies $L_2$ normalization along last dimension (`dim=-1`).
- **`compute_zero_shot_logits(images, candidate_labels)`**:
  - Direct zero-shot probability extraction applying model scale and bias:
    $$P(I, T) = \sigma(\text{logits\_per\_image})$$

---

### 4.3. `src/video_search.py` (`VideoSearchEngine`)

In-memory vector store and mathematical similarity engine.

#### Internal State:
- `self.frame_embeddings`: Float tensor shape $(N, D)$, where $N$ = total frames sampled, $D$ = embedding dimension (768 for `base`, 1152 for `so400m`).
- `self.frame_metadata`: List of length $N$ containing metadata dictionaries `{"timestamp": float, "frame_idx": int, "video_path": str}`.

#### Retrieval Workflows:

#### A. Text-to-Moment Search (`search_moments`)
1. Query string embedded to normalized vector $\mathbf{q} \in \mathbb{R}^{1 \times D}$.
2. Cosine similarity computed by single matrix multiplication:
   $$\mathbf{S} = \mathbf{E}_{\text{frames}} \mathbf{q}^\top \in \mathbb{R}^{N \times 1}$$
   *(Embeddings pre-normalized $\rightarrow$ dot product equals cosine similarity).*
3. `torch.topk(similarities, k=top_k)` extracts $k$ largest values and their row indices.
4. Outputs list of `SearchResult(timestamp, frame_idx, score, video_path)`.

#### B. Zero-Shot Video Classification (`classify_video_zero_shot`)
1. Text labels vector encoded to $\mathbf{T} \in \mathbb{R}^{M \times D}$, where $M$ is label count.
2. Full video-frame similarity matrix:
   $$\mathbf{M}_{\text{sim}} = \mathbf{E}_{\text{frames}} \mathbf{T}^\top \in \mathbb{R}^{N \times M}$$
3. Temporal aggregations per label $j \in [1, M]$:
   - **Mean score** (sustained presence across duration):
     $$\bar{s}_j = \frac{1}{N} \sum_{i=1}^N \mathbf{M}_{ij}$$
   - **Peak score** (maximum single-frame occurrence):
     $$\hat{s}_j = \max_{1 \le i \le N} \mathbf{M}_{ij}$$
4. Ranked descending by `mean_score`.

---

## 5. Endpoints & Interfaces

### 5.1. CLI (`main.py`)

Provides headless execution and synthetic validation video generation.

```bash
# Generate synthetic video and run default query
python3 main.py --generate-dummy

# Query custom video
python3 main.py --video sample_data/demo.mp4 --query "blue ocean with water" --fps 1.0

# Zero-shot classification
python3 main.py --video sample_data/demo.mp4 --labels "red scene,green field,ocean"
```

#### Synthetic Generator (`generate_dummy_video`):
- Writes 6-second $320 \times 240$ video at 20 FPS using OpenCV.
- Splits video into 3 distinct RGB blocks labeled with text:
  1. Red block (`"red scene with an apple"`, BGR `(0, 0, 200)`)
  2. Green block (`"green field with trees"`, BGR `(0, 200, 0)`)
  3. Blue block (`"blue ocean with water"`, BGR `(200, 100, 0)`)

---

### 5.2. Web Interface (`app.py`)

Gradio dashboard running locally on port `7860`.

```
┌────────────────────────────────────────────────────────────────────────┐
│                      SigLIP Video Intelligence Lab                     │
├─────────────────────────┬──────────────────────────────────────────────┤
│ CONFIGURATION           │ OUTPUT TABS                                  │
│                         │                                              │
│ - Video File Input      │ [🔍 Text-to-Moment Search]                   │
│ - Model Selector        │   - Query Textbox & Top-K Slider             │
│   (base-224 / so400m)   │   - Search Action Button                     │
│ - Sampling Rate (FPS)   │   - Frame Matches Image Gallery              │
│                         │   - Detailed Ranking Table (DataFrame)       │
│                         │   - Auto-Extracted Highlight Clip (±1.5s)    │
│                         │                                              │
│                         │ [🏷️ Zero-Shot Classification]                │
│                         │   - Comma-separated Labels Textbox           │
│                         │   - Classification Table (Mean & Peak score) │
└─────────────────────────┴──────────────────────────────────────────────┘
```

#### Optimization Features:
- **`_engine_cache`**: Dict cache prevents re-instantiating heavy neural weights across successive queries.
- **Dynamic Clip Preview**: Slices $[t_{\text{best}} - 1.5s, t_{\text{best}} + 1.5s]$ around top-ranked timestamp and mounts MP4 directly into Gradio video player.

---

### 5.3. Test Suite (`tests/test_pipeline.py`)

Automated validation using Python `unittest`:
- Generates temporal two-color test video fixture (`test_dummy.mp4`).
- Asserts `VideoProcessor` frame extraction count, schema, timestamp validity, and clipping capability.
- Validates `SigLIPEngine` and `VideoSearchEngine` indexing, shape integrity, top-k ordering, and classification outputs.
- Cleans up generated artifacts upon test teardown.

---

## 6. Execution Flow Diagram

```
User Input Query / Labels
          │
          ▼
┌───────────────────┐
│     main.py /     │
│      app.py       │
└─────────┬─────────┘
          │ (1) Pass video path + target FPS
          ▼
┌────────────────────────────────────────────────────────┐
│ VideoProcessor.extract_frames()                        │
│   - cv2.VideoCapture extracts frames at sample rate    │
│   - Converts BGR -> RGB -> PIL Image                   │
│   - Tracks exact timestamp: frame_idx / native_fps     │
└─────────────────────────┬──────────────────────────────┘
                          │ (2) List of PIL Images
                          ▼
┌────────────────────────────────────────────────────────┐
│ SigLIPEngine.encode_images()                           │
│   - AutoProcessor scales & converts to PyTorch tensors │
│   - Model forward pass -> Vision Transformer features  │
│   - L2 Normalization: F.normalize(..., p=2)            │
└─────────────────────────┬──────────────────────────────┘
                          │ (3) Matrix E: Shape (N, D)
                          ▼
┌────────────────────────────────────────────────────────┐
│ VideoSearchEngine.index_video()                        │
│   - Stores frame embeddings tensor                     │
│   - Stores metadata dicts (timestamp, frame_idx)       │
└─────────────────────────┬──────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
[ Text-to-Moment Search ]       [ Zero-Shot Classification ]
         │                                 │
         │ Encode Query text               │ Encode M Candidate Labels
         │ q: (1, D)                       │ T: (M, D)
         ▼                                 ▼
   Matrix Product:                   Matrix Product:
   S = E @ q.T                       M_sim = E @ T.T
   (Shape: N)                        (Shape: N x M)
         │                                 │
         ▼                                 ▼
   torch.topk(S, k)                  mean(dim=0) & max(dim=0)
         │                                 │
         ▼                                 ▼
   SearchResult List                 Ranked Category Scores
   - Timestamp                       - Mean Presence
   - Frame Index                     - Peak Moment
   - Cosine Score
```

---

## 7. Performance Characteristics & System Bounds

1. **Embedding Storage Complexity**:
   - For 1 hour video at 1.0 FPS: $N = 3600$ frames.
   - For `google/siglip-base-patch16-224` ($D=768$, `float32`):
     $$\text{Size} = 3600 \times 768 \times 4\text{ bytes} \approx 11.06\text{ MB}$$
   - Allows fast, fully in-memory vector search without external vector DB for standard videos.
2. **Computational Bottleneck & Query Latency**:
   - Frame extraction and vision transformer forward passes represent $>95\%$ of indexing runtime (~1.31s / frame on CPU).
   - While the raw vector dot product similarity ranking alone executes in $<2\text{ms}$ on CPU once indexed, true end-to-end query latency includes text tokenization and text transformer forward pass (~100–260 ms on CPU). In empirical benchmarks (`evaluate.py`), average end-to-end query latency measures ~477.54 ms (reflecting CPU text encoding ~263.27 ms + ranking overhead).

---

## 8. Empirical Benchmark Evaluation & Verification

Benchmark evaluations are executed via `evaluate.py` using `google/siglip-base-patch16-224` on CPU (`torch.float32`), tested on an 8-scene synthetic multi-domain video (24.0s duration, 24 natural queries, 4 negative distractors) under clean visual settings (`clean_no_overlay`, `--with-text-overlay=False`).

### 8.1. Text-to-Moment Retrieval Performance

| Metric | Measured Value | Description |
|---|---|---|
| **Recall@1 (R@1)** | **87.50%** (21/24) | Proportion of queries where top-1 timestamp falls within target half-open interval $[t_{\text{start}}, t_{\text{end}})$ |
| **Recall@3 (R@3)** | **91.67%** (22/24) | Target interval hit within top-3 predicted timestamps |
| **Recall@5 (R@5)** | **91.67%** (22/24) | Target interval hit within top-5 predicted timestamps |
| **Mean Reciprocal Rank (MRR)** | **0.8958** | Mean reciprocal rank of the first correct scene interval match |
| **Mean Absolute Error (MAE to Interval)** | **1.12s** | Mean temporal distance to the scene interval $[t_{\text{start}}, t_{\text{end}})$ (0.0s if inside) |
| **Mean Absolute Error (MAE to Center)** | **2.08s** | Mean absolute temporal distance to the symmetric center of the target scene |
| **Hit Rate ($\Delta t_{\text{interval}} = 0.0\text{s}$)** | **87.50%** (21/24) | Predicted timestamp falls strictly within target scene interval |
| **Hit Rate ($\Delta t_{\text{interval}} \le 1.0\text{s}$)** | **87.50%** (21/24) | Distance to target scene interval is within 1.0 second |
| **Hit Rate ($\Delta t_{\text{interval}} \le 2.0\text{s}$)** | **87.50%** (21/24) | Distance to target scene interval is within 2.0 seconds |
| **Hit Rate ($\Delta t_{\text{center}} \le 1.0\text{s}$)** | **54.17%** (13/24) | Temporal distance to scene center is within 1.0 second |
| **Hit Rate ($\Delta t_{\text{center}} \le 2.0\text{s}$)** | **87.50%** (21/24) | Temporal distance to scene center is within 2.0 seconds |
| **Average Query Latency** | **477.54 ms** | End-to-end CPU query latency (text encoding ~263.27 ms + vector similarity ranking) |

### 8.2. Zero-Shot Video Classification

Evaluated across 12 candidate classes (8 ground-truth scenes + 4 negative distractors):

| Metric | Measured Value | Description |
|---|---|---|
| **Scene Top-1 Accuracy** | **100.00%** (8/8) | All test scenes correctly identified by top-1 category |
| **Scene Top-3 Accuracy** | **100.00%** (8/8) | Ground-truth class present in top-3 candidates |
| **Average True Label Score** | **-0.0357** | Mean cosine similarity for ground truth categories |
| **Average Distractor Score** | **-0.0499** | Mean cosine similarity for distractor categories |
| **Semantic Separation Margin** | **+0.0142** | Mean cosine separation margin between true and distractor labels |
| **Native SigLIP Sigmoid Probability (True)** | **0.58%** (0.0058) | Calibrated pairwise sigmoid probability $\sigma(\tau \mathbf{u}^\top \mathbf{v} + b)$ per scene |
| **Native SigLIP Sigmoid Probability (False)** | **0.00%** (0.0000) | Distractor categories fully suppressed to 0.00% |
| **Global Video True Probability** | **0.04%** (0.0004) | Mean calibrated sigmoid probability over entire video |
| **Global Video Distractor Probability** | **0.00%** (0.0000) | Mean calibrated distractor probability over entire video |
| **Global Probability Separation Margin** | **+0.04%** (0.0004) | Calibrated margin on whole-video aggregation |

### 8.3. FPS Sampling Rate Ablation Study

Sensitivity across frame extraction rates on 24.0s benchmark video:

| Target FPS | Frames Indexed | Indexing Time | Throughput | Recall@1 | MAE to Interval | MAE to Center |
|---|---|---|---|---|---|---|
| **0.5 FPS** | 12 frames | 11.09s (ext: 0.14s, enc: 10.96s) | 1.08 frames/s | **87.50%** | **1.08s** | **1.96s** |
| **1.0 FPS** | 24 frames | 26.11s (ext: 0.16s, enc: 25.95s) | 0.92 frames/s | **87.50%** | **1.12s** | **2.08s** |
| **2.0 FPS** | 48 frames | 52.31s (ext: 0.33s, enc: 51.98s) | 0.92 frames/s | **100.00%** | **0.00s** | **1.19s** |

### 8.4. Historical Note on OCR Text Leakage Artifact

> **Historical Note (OCR Text Leakage vs. Clean Visual Benchmark):**
> Initial exploratory reports recorded 100.00% Recall@1 and 100.00% Hit Rate across all sampling rates (including 0.5 FPS). Auditing the synthetic generator revealed that previous test videos drew category names (`sc["category"]`) directly onto synthetic frames via `cv2.putText()`. Because SigLIP incorporates multimodal OCR capabilities learned during pre-training on web image-text pairs, the model effectively read the text labels rather than performing pure visual grounding.
> 
> With the introduction of `--with-text-overlay=False` (`clean_no_overlay` mode), benchmark frames contain only raw visual content. The honest visual retrieval performance is measured at **87.50% R@1** at 0.5 FPS and 1.0 FPS, scaling to **100.00% R@1** at 2.0 FPS. The OCR leakage baseline can be explicitly reproduced using the `--with-text-overlay` flag in `evaluate.py`.
