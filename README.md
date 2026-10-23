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

### 4. Tìm kiếm khoảnh khắc trong video thực tế
```bash
python3 main.py --video path/to/your_video.mp4 --query "a person driving a car" --fps 1.0
```

### 5. Zero-shot phân loại video
```bash
python3 main.py --video path/to/your_video.mp4 --labels "action movie,sports game,cooking tutorial"
```

### 6. Phân loại video với Linear Adapter đã huấn luyện
```bash
python3 main.py --video sample_data/demo.mp4 --adapter-checkpoint checkpoints/adapter_msad_base224.pt --labels "fighting,theft"
```

### 7. Chạy đánh giá toàn diện mô hình (Full Evaluation Benchmark)
```bash
python3 evaluate.py
```

### 8. Đánh giá trên dữ liệu camera giám sát thực tế (Real Surveillance Benchmark)
```bash
python3 evaluate_real.py
```

## Kết quả đánh giá mô hình toàn diện (Full Model Evaluation)

Thực nghiệm đo lường hiệu năng trên mô hình `google/siglip-base-patch16-224` (203.2M tham số) với video benchmark đa miền gồm 8 cảnh trực quan khác biệt, 24 truy vấn mô tả ngôn ngữ tự nhiên và 4 nhãn âm tính (distractors), sử dụng dữ liệu thị giác thực tế không có chữ chèn (`clean_no_overlay`).

### 1. Hiệu năng truy xuất khoảnh khắc (Text-to-Moment Retrieval)

| Metric | Giá trị | Giải thích |
|---|---|---|
| **Recall@1 (R@1)** | **87.50%** (21/24) | Tỷ lệ timestamp dự đoán top-1 rơi trúng đoạn video mục tiêu $[t_{\text{start}}, t_{\text{end}})$ |
| **Recall@3 (R@3)** | **91.67%** (22/24) | Tỷ lệ tìm thấy đoạn mục tiêu trong top 3 kết quả |
| **Recall@5 (R@5)** | **91.67%** (22/24) | Tỷ lệ tìm thấy đoạn mục tiêu trong top 5 kết quả |
| **Mean Reciprocal Rank (MRR)** | **0.8958** | Thứ hạng nghịch đảo trung bình của kết quả đúng |
| **Mean Absolute Error (MAE to Interval)** | **1.12s** | Độ lệch thời gian trung bình đến khoảng $[t_{\text{start}}, t_{\text{end}})$ (0.0s nếu trúng trong phân đoạn) |
| **Mean Absolute Error (MAE to Center)** | **2.08s** | Độ lệch thời gian trung bình so với tâm đối xứng của phân đoạn |
| **Hit Rate ($\Delta t_{\text{interval}} = 0.0\text{s}$)** | **87.50%** (21/24) | Tỷ lệ timestamp dự đoán rơi chính xác vào phân đoạn mục tiêu |
| **Hit Rate ($\Delta t_{\text{interval}} \le 1.0\text{s}$)** | **87.50%** (21/24) | Tỷ lệ khoảng cách tới đoạn mục tiêu $\le 1.0$ giây |
| **Hit Rate ($\Delta t_{\text{interval}} \le 2.0\text{s}$)** | **87.50%** (21/24) | Tỷ lệ khoảng cách tới đoạn mục tiêu $\le 2.0$ giây |
| **Hit Rate ($\Delta t_{\text{center}} \le 1.0\text{s}$)** | **54.17%** (13/24) | Tỷ lệ sai số so với tâm đoạn clip $\le 1.0$ giây |
| **Hit Rate ($\Delta t_{\text{center}} \le 2.0\text{s}$)** | **87.50%** (21/24) | Tỷ lệ sai số so với tâm đoạn clip $\le 2.0$ giây |
| **Độ trễ trung bình mỗi truy vấn** | **477.54 ms** | Đo lường thực tế end-to-end trên CPU (gồm text encoding ~263.27 ms và ranking ma trận) |

#### Một số kết quả truy vấn mẫu:

| Cảnh mục tiêu | Câu truy vấn tự nhiên | Timestamp tìm được | Tâm cảnh thực | Sai số khoảng / tâm | Điểm tương đồng |
|---|---|---|---|---|---|
| Red apple in fruit basket | *"a bright red apple"* | 1.00s | 1.50s | 0.00s / 0.50s | 0.0478 |
| Red apple in fruit basket | *"fresh red fruit"* | 0.00s | 1.50s | 0.00s / 1.50s | 0.0265 |
| Red apple in fruit basket | *"crimson red object"* | 0.00s | 1.50s | 0.00s / 1.50s | 0.0886 |
| Green forest trees | *"green forest trees"* | 3.00s | 4.50s | 0.00s / 1.50s | 0.0395 |
| Green forest trees | *"vibrant green foliage"* | 3.00s | 4.50s | 0.00s / 1.50s | 0.0484 |
| Green forest trees | *"nature woodland greenery"* | 3.00s | 4.50s | 0.00s / 1.50s | 0.0521 |
| Blue ocean water | *"blue ocean water"* | 6.00s | 7.50s | 0.00s / 1.50s | 0.0705 |
| Blue ocean water | *"deep sea waves"* | 6.00s | 7.50s | 0.00s / 1.50s | 0.0840 |

---

### 2. Phân loại Zero-shot (Zero-Shot Video Classification)

Đánh giá trên không gian 12 lớp (8 lớp cảnh thực + 4 lớp đánh lạc hướng / negative distractors):

| Chỉ số | Kết quả | Ghi chú |
|---|---|---|
| **Scene Top-1 Accuracy** | **100.00%** (8/8) | Tất cả các phân đoạn đều nhận dạng đúng lớp top-1 |
| **Scene Top-3 Accuracy** | **100.00%** (8/8) | Lớp chuẩn luôn nằm trong top 3 dự đoán |
| **Average True Label Score** | **-0.0357** | Điểm tương đồng Cosine trung bình của các nhãn thực tế |
| **Average Distractor Score** | **-0.0499** | Điểm tương đồng Cosine trung bình của các nhãn gây nhiễu |
| **Semantic Separation Margin** | **+0.0142** | Khoảng cách phân tách ngữ nghĩa Cosine giữa nhãn thật và nhãn giả |
| **Native SigLIP Sigmoid Probability (True - Scene)** | **0.58%** | Xác suất Sigmoid cục bộ theo phân cảnh ($\sigma(\tau \mathbf{u}^\top \mathbf{v} + b)$) |
| **Native SigLIP Sigmoid Probability (False - Scene)** | **0.00%** | Các nhãn âm tính bị triệt tiêu hoàn toàn về 0 |
| **Native SigLIP Sigmoid Probability (True - Global Video)** | **0.04%** | Xác suất Sigmoid trung bình trên toàn bộ video |
| **Native SigLIP Sigmoid Probability (False - Global Video)** | **0.00%** | Nhãn âm tính trên toàn bộ video triệt tiêu về 0 |
| **Global Probability Separation Margin** | **+0.04%** | Khoảng cách phân tách xác suất toàn cục |

---

### 3. Nghiên cứu độ nhạy tần số lấy mẫu (FPS Sampling Rate Ablation)

So sánh giữa các mức FPS trích xuất trên video benchmark độ dài 24.0s:

| Target FPS | Số Frame trích xuất | Thời gian Indexing | Tốc độ xử lý (FPS) | Recall@1 | Sai số khoảng (MAE Interval) | Sai số tâm (MAE Center) |
|---|---|---|---|---|---|---|
| **0.5 FPS** | 12 frames | 11.09s (trích xuất 0.14s, mã hoá 10.96s) | 1.08 frames/s | **87.50%** | **1.08s** | **1.96s** |
| **1.0 FPS** | 24 frames | 26.11s (trích xuất 0.16s, mã hoá 25.95s) | 0.92 frames/s | **87.50%** | **1.12s** | **2.08s** |
| **2.0 FPS** | 48 frames | 52.31s (trích xuất 0.33s, mã hoá 51.98s) | 0.92 frames/s | **100.00%** | **0.00s** | **1.19s** |

> **Nhận xét thực nghiệm**: Mức lấy mẫu **1.0 FPS** mang lại sự cân bằng tối ưu giữa thời gian lập chỉ mục (~26.1s trên CPU) và độ chính xác phân loại/định vị ($R@1 = 87.50\%$). Khi cần độ chính xác tuyệt đối không sai lệch biên ($R@1 = 100.00\%, \text{MAE}_{\text{interval}} = 0.00\text{s}$), mức **2.0 FPS** là lựa chọn phù hợp với chi phí tính toán tăng gấp đôi.

---

### 4. Tài nguyên & Thông số phần cứng kiểm thử

- **Model**: `google/siglip-base-patch16-224`
- **Số lượng tham số**: 203,155,970 (~203.2M params)
- **Logit Scale ($\tau$)**: 117.33 | **Logit Bias ($b$)**: -12.93
- **Môi trường thực thi**: CPU (torch.float32)
- **Tốc độ trích xuất khung hình**: ~7.49 ms / frame (~133.42 frames/giây)
- **Tốc độ mã hoá hình ảnh**: ~1312.88 ms / frame (~0.76 frames/giây trên CPU)
- **Tốc độ mã hoá văn bản**: ~263.27 ms / query
- **Độ trễ truy vấn end-to-end trung bình**: ~477.54 ms (thực tế đo lường gồm text encoding + vector similarity ranking; phép tính nhân ma trận vector đơn thuần mất $<2$ ms)

> ### ⚠️ Ghi chú lịch sử về hiện tượng rò rỉ nhãn văn bản (Historical Note on OCR Text Leakage)
> Trong báo cáo sơ bộ ban đầu, hệ thống ghi nhận kết quả Recall@1 đạt **100.00%** và Hit Rate đạt **100.00%** ngay cả ở tần số thấp 0.5 FPS. Qua kiểm toán mã nguồn thực nghiệm, nguyên nhân của con số 100% hoàn hảo này đến từ **hiện tượng rò rỉ nhãn OCR**: hàm sinh video giả lập ban đầu đã dùng `cv2.putText()` vẽ trực tiếp chuỗi tên danh mục (`category`) bằng chữ lớn lên bề mặt frame. Vì mô hình SigLIP được huấn luyện trên cặp hình ảnh - văn bản web đa phương thức có khả năng nhận dạng ký tự (OCR), mô hình đã trực tiếp "đọc" chữ viết trên ảnh thay vì trích xuất đặc trưng thị giác (màu sắc, cấu trúc không gian).
> 
> Sau khi khắc phục bằng chế độ video chuẩn không nhãn chữ (`clean_no_overlay`, `--with-text-overlay=False`), các chỉ số được đo lường trung thực: Recall@1 đạt **87.50%** tại 0.5 & 1.0 FPS và đạt **100.00%** khi lấy mẫu dày hơn tại 2.0 FPS. Hiện tượng rò rỉ nhãn OCR này đã được kiểm chứng đối chứng thông qua cờ `--with-text-overlay` (OCR baseline control) trong script `evaluate.py`.

## Điểm khác biệt kỹ thuật của SigLIP so với CLIP

- **Mất mát Sigmoid (BCE Loss)**: Thay vì dùng Softmax chuẩn hoá trên toàn bộ batch (đòi hỏi batch size cực lớn và global communication), SigLIP coi mỗi cặp ảnh-văn bản là một bài toán phân loại nhị phân độc lập.
- **Hiệu quả cao khi scale**: Tối ưu tốc độ huấn luyện và bộ nhớ, chất lượng embedding mạnh hơn CLIP ở cùng quy mô tham số.

---

## Đánh giá trên dữ liệu Camera Giám sát Thực tế (Real-World Surveillance Evaluation)

Đánh giá thực nghiệm hiệu năng của mô hình SigLIP trên tập dữ liệu camera giám sát thực tế MSAD (240 video, 5,331 khung hình tại 1.0 FPS) qua 11 lớp hành vi/bất thường an ninh, sử dụng phương pháp 5-Fold Stratified Group K-Fold (chia nhóm theo góc máy camera nhằm tránh rò rỉ dữ liệu nền).

### 1. So sánh Macro-F1: Zero-Shot vs. Prompt-Ensemble vs. Linear Adapter

| Phương pháp | Macro-F1 | Mức tăng so với Zero-Shot | Đánh giá |
|---|---|---|---|
| **Zero-Shot Baseline** | **12.53%** (0.1253) | — | Phân loại trực tiếp bằng cosine similarity giữa video embedding và tên nhãn |
| **Prompt-Ensemble** | **23.38%** (0.2338) | **+10.85%** (+0.1085) | Trung bình L2-normalized embedding qua 4 mẫu prompt giám sát chuẩn |
| **Linear Adapter (5-fold OOF)** | **19.99%** (0.1999) | **+7.46%** (+0.0746) | Đầu phân loại tuyến tính huấn luyện trên vector video mean-pooled |

- **Kiểm định thống kê Bootstrap ghép cặp (Paired Bootstrap 95% CI)**:
  - Khoảng tin cậy 95% của độ chênh lệch Macro-F1 $(\text{Adapter} - \text{Zero-Shot})$: **[0.0242, 0.1232]** ($[+2.42\%, +12.32\%]$).
  - Vì ngưỡng dưới $0.0242 > 0$, mức cải thiện mang ý nghĩa thống kê ở mức $\alpha = 0.05$.
  - **Kết luận chính thức (Formal Verdict)**: `gain` (xác nhận cải thiện có ý nghĩa thống kê vượt trội so với baseline Zero-Shot).

### 2. Chi tiết hiệu năng theo từng lớp bất thường (Per-Class F1 Score)

| Lớp bất thường (MSAD 11 Classes) | Zero-Shot F1 | Prompt-Ensemble F1 | Linear Adapter F1 |
|---|---|---|---|
| **Assault** | 0.00% (0.0000) | 0.00% (0.0000) | 0.00% (0.0000) |
| **Explosion** | 0.00% (0.0000) | 28.57% (0.2857) | 0.00% (0.0000) |
| **Fighting** | 14.29% (0.1429) | 0.00% (0.0000) | 0.00% (0.0000) |
| **Fire** | 21.43% (0.2143) | 57.14% (0.5714) | 48.65% (0.4865) |
| **Object_falling** | 8.00% (0.0800) | 18.18% (0.1818) | 0.00% (0.0000) |
| **People_falling** | 8.51% (0.0851) | 23.53% (0.2353) | 52.35% (0.5235) |
| **Robbery** | 28.43% (0.2843) | 37.91% (0.3791) | 67.53% (0.6753) |
| **Shooting** | 0.00% (0.0000) | 0.00% (0.0000) | 0.00% (0.0000) |
| **Traffic_accident** | 26.42% (0.2642) | 33.33% (0.3333) | 51.33% (0.5133) |
| **Vandalism** | 0.00% (0.0000) | 24.00% (0.2400) | 0.00% (0.0000) |
| **Water_incident** | 30.77% (0.3077) | 34.48% (0.3448) | 0.00% (0.0000) |
| **Macro-F1 Trung bình** | **12.53% (0.1253)** | **23.38% (0.2338)** | **19.99% (0.1999)** |

### 3. Thử nghiệm chuyển giao liên miền trên tập STEC (STEC Cross-Domain Transfer Probe)

Đánh giá thăm dò trên 14 video camera giám sát thực tế từ tập dữ liệu STEC nhằm kiểm nghiệm khả năng khái quát hoá liên miền của mô hình sang môi trường camera giao thông/đô thị tại Việt Nam.

> **Tuyên bố giới hạn mẫu nhỏ (Small-Sample Disclaimer)**:
> Kích thước mẫu thử nghiệm rất nhỏ ($n < 10$ cho mỗi lớp, tổng $n = 14$ video, trong đó 11 video ánh xạ được sang các lớp MSAD tương ứng). Kết quả chỉ mang tính chất thăm dò khám phá (exploratory probe) và **không có ý nghĩa khẳng định về mặt thống kê**.

| Hành vi (STEC) | Lớp ánh xạ (MSAD) | Số mẫu ($n$) | Zero-Shot Accuracy (Wilson 95% CI) | Linear Adapter Accuracy (Wilson 95% CI) |
|---|---|---|---|---|
| `fighting` | Fighting | 5 | 0.00% (0/5) [0.00%, 43.45%] | 0.00% (0/5) [0.00%, 43.45%] |
| `people_falling` | People_falling | 4 | 0.00% (0/4) [0.00%, 48.99%] | 25.00% (1/4) [4.56%, 69.94%] |
| `theft` | Robbery | 2 | 100.00% (2/2) [34.24%, 100.00%] | 0.00% (0/2) [0.00%, 65.76%] |
| **Tổng các lớp ánh xạ** | — | **11** | **18.18% (2/11)** | **9.09% (1/11)** |

### 4. Thời gian thực thi & Hiệu năng phần cứng (Hardware Runtime & Latency)

- **Thời gian trích xuất đặc trưng**: ~5.7s / video (~0.27s / frame trên CPU với `torch.float32`; tổng thời gian xử lý 5,331 frames của 240 video là ~1,439s).
- **Thời gian huấn luyện Linear Adapter**: **~5.0s** cho toàn bộ 240 video trên CPU (200 epochs tối ưu hóa L-BFGS trên các vector đặc trưng đã cache).
- **Độ trễ suy luận (Inference Latency)**: **< 2 ms** trên mỗi video (thực hiện phép nhân ma trận qua linear head trên vector embedding mean-pooled).

> **Tuân thủ quyền riêng tư (Privacy Compliance Guardrail)**: Báo cáo này tuyệt đối không lưu trữ, trích xuất hoặc hiển thị bất kỳ khung hình video, ảnh chụp màn hình hay khuôn mặt cá nhân nào từ các tập dữ liệu camera giám sát thực tế (MSAD, STEC).
