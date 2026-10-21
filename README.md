# Mini Project: Video Search & Zero-shot Classification with SigLIP

Dự án mini ứng dụng mô hình SigLIP (Sigmoid Loss for Language Image Pre-training) vào xử lý video:
1. **Text-to-Moment Retrieval**: Tìm timestamp/đoạn video khớp với câu mô tả văn bản tự nhiên.
2. **Zero-Shot Video Classification**: Phân loại hành động/nội dung video mà không cần huấn luyện lại.

## Cấu trúc thư mục

```
NCKH/
├── src/
│   ├── video_processor.py   # Trích xuất frame theo FPS & timestamp, trích đoạn clip
│   ├── siglip_engine.py     # Wrapper SigLIP (AutoProcessor, AutoModel)
│   └── video_search.py      # Bộ máy tìm kiếm và phân loại video
├── tests/
│   └── test_pipeline.py     # Unit test tự động
├── sample_data/             # Chứa video mẫu & benchmark
├── evaluate.py              # Script chạy full benchmark đánh giá mô hình
├── eval_results.json        # Dữ liệu kết quả benchmark chi tiết dạng JSON
├── app.py                   # Giao diện Web trực quan Gradio
├── main.py                  # Script CLI chạy thử nghiệm
└── requirements.txt
```

## Hướng dẫn cài đặt & sử dụng

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Giao diện Web (Gradio)
```bash
python3 app.py
```
Truy cập `http://localhost:7860` trên trình duyệt để upload video, tìm kiếm khoảnh khắc theo text, xem gallery ảnh khớp và xem trước clip cắt tự động quanh timestamp.

### 3. Chạy thử nghiệm với video giả lập tự sinh (dummy video) qua CLI
```bash
python3 main.py --generate-dummy
```

### 3. Tìm kiếm khoảnh khắc trong video thực tế
```bash
python3 main.py --video path/to/your_video.mp4 --query "a person driving a car" --fps 1.0
```

### 4. Zero-shot phân loại video
```bash
python3 main.py --video path/to/your_video.mp4 --labels "action movie,sports game,cooking tutorial"
```

### 5. Chạy đánh giá toàn diện mô hình (Full Evaluation Benchmark)
```bash
python3 evaluate.py
```

## Kết quả đánh giá mô hình toàn diện (Full Model Evaluation)

Thực nghiệm đo lường hiệu năng trên mô hình `google/siglip-base-patch16-224` (203.2M tham số) với video benchmark đa miền gồm 8 cảnh trực quan khác biệt, 24 truy vấn mô tả ngôn ngữ tự nhiên và 4 nhãn âm tính (distractors).

### 1. Hiệu năng truy xuất khoảnh khắc (Text-to-Moment Retrieval)

| Metric | Giá trị | Giải thích |
|---|---|---|
| **Recall@1 (R@1)** | **100.00%** (24/24) | Tỷ lệ timestamp dự đoán top-1 rơi trúng đoạn video mục tiêu |
| **Recall@3 (R@3)** | **100.00%** (24/24) | Tỷ lệ tìm thấy đoạn mục tiêu trong top 3 kết quả |
| **Recall@5 (R@5)** | **100.00%** (24/24) | Tỷ lệ tìm thấy đoạn mục tiêu trong top 5 kết quả |
| **Mean Reciprocal Rank (MRR)** | **1.0000** | Thứ hạng nghịch đảo trung bình của kết quả đúng |
| **Mean Absolute Error (MAE)** | **0.92s** | Độ lệch thời gian trung bình giữa timestamp tìm được và tâm đoạn clip |
| **Hit Rate ($\Delta t \le 1.0\text{s}$)** | **58.33%** | Tỷ lệ kết quả trúng đích trong khoảng dung sai $\pm 1.0$ giây |
| **Hit Rate ($\Delta t \le 2.0\text{s}$)** | **100.00%** | Tỷ lệ kết quả trúng đích trong khoảng dung sai $\pm 2.0$ giây |
| **Độ trễ trung bình mỗi truy vấn** | **177.50 ms** | Bao gồm text tokenization, text encoding và vector similarity ranking |

#### Một số kết quả truy vấn mẫu:

| Cảnh mục tiêu | Câu truy vấn tự nhiên | Timestamp tìm được | Tâm cảnh thực | Sai số thời gian | Điểm tương đồng |
|---|---|---|---|---|---|
| Red apple in fruit basket | *"a bright red apple"* | 0.00s | 1.50s | 1.50s | 0.1106 |
| Red apple in fruit basket | *"fresh red fruit"* | 0.00s | 1.50s | 1.50s | 0.0809 |
| Green forest trees | *"green forest trees"* | 5.00s | 4.50s | 0.50s | 0.0943 |
| Green forest trees | *"vibrant green foliage"* | 5.00s | 4.50s | 0.50s | 0.0608 |
| Blue ocean water | *"blue ocean water"* | 6.00s | 7.50s | 1.50s | 0.1135 |
| Blue ocean water | *"deep sea waves"* | 6.00s | 7.50s | 1.50s | 0.1168 |

---

### 2. Phân loại Zero-shot (Zero-Shot Video Classification)

Đánh giá trên không gian 12 lớp (8 lớp cảnh thực + 4 lớp đánh lạc hướng / negative distractors):

| Chỉ số | Kết quả | Ghi chú |
|---|---|---|
| **Scene Top-1 Accuracy** | **100.00%** (8/8) | Tất cả các phân đoạn đều nhận dạng đúng lớp top-1 |
| **Scene Top-3 Accuracy** | **100.00%** (8/8) | Lớp chuẩn luôn nằm trong top 3 dự đoán |
| **Average True Label Score** | **-0.0341** | Điểm tương đồng Cosine trung bình của các nhãn thực tế |
| **Average Distractor Score** | **-0.0558** | Điểm tương đồng Cosine trung bình của các nhãn gây nhiễu |
| **Semantic Separation Margin** | **+0.0218** | Khoảng cách phân tách ngữ nghĩa giữa nhãn thật và nhãn giả |
| **Native SigLIP Sigmoid Probability (True)** | **~7.50%** | Xác suất Sigmoid sau hiệu chuẩn ($\sigma(\tau \mathbf{u}^\top \mathbf{v} + b)$) |
| **Native SigLIP Sigmoid Probability (False)** | **0.00%** | Các nhãn âm tính bị triệt tiêu hoàn toàn về 0 |

---

### 3. Nghiên cứu độ nhạy tần số lấy mẫu (FPS Sampling Rate Ablation)

So sánh giữa các mức FPS trích xuất trên video benchmark độ dài 24.0s:

| Target FPS | Số Frame trích xuất | Thời gian Indexing | Tốc độ xử lý (FPS) | Recall@1 | Sai số thời gian (MAE) |
|---|---|---|---|---|---|
| **0.5 FPS** | 12 frames | 4.25s | 2.82 frames/s | **100.0%** | **0.79s** |
| **1.0 FPS** | 24 frames | 8.06s | 2.98 frames/s | **100.0%** | **0.92s** |
| **2.0 FPS** | 48 frames | 16.85s | 2.85 frames/s | **100.0%** | **0.90s** |

> **Nhận xét thực nghiệm**: Mức lấy mẫu **0.5 FPS - 1.0 FPS** là điểm cân bằng tối ưu giữa độ chính xác định vị thời gian ($< 1.0\text{s}$) và chi phí tính toán / thời gian trích xuất embedding.

---

### 4. Tài nguyên & Thông số phần cứng kiểm thử

- **Model**: `google/siglip-base-patch16-224`
- **Số lượng tham số**: 203,155,970 (~203.2M params)
- **Logit Scale ($\tau$)**: 117.33 | **Logit Bias ($b$)**: -12.93
- **Môi trường thực thi**: CPU (torch.float32)
- **Tốc độ mã hoá hình ảnh**: ~356.41 ms / frame (~2.81 frames/giây trên CPU)
- **Tốc độ mã hoá văn bản**: ~103.15 ms / query

## Điểm khác biệt kỹ thuật của SigLIP so với CLIP

- **Mất mát Sigmoid (BCE Loss)**: Thay vì dùng Softmax chuẩn hoá trên toàn bộ batch (đòi hỏi batch size cực lớn và global communication), SigLIP coi mỗi cặp ảnh-văn bản là một bài toán phân loại nhị phân độc lập.
- **Hiệu quả cao khi scale**: Tối ưu tốc độ huấn luyện và bộ nhớ, chất lượng embedding mạnh hơn CLIP ở cùng quy mô tham số.
