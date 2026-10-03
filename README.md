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
├── sample_data/             # Chứa video mẫu
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

## Điểm khác biệt kỹ thuật của SigLIP so với CLIP

- **Mất mát Sigmoid (BCE Loss)**: Thay vì dùng Softmax chuẩn hoá trên toàn bộ batch (đòi hỏi batch size cực lớn và global communication), SigLIP coi mỗi cặp ảnh-văn bản là một bài toán phân loại nhị phân độc lập.
- **Hiệu quả cao khi scale**: Tối ưu tốc độ huấn luyện và bộ nhớ, chất lượng embedding mạnh hơn CLIP ở cùng quy mô tham số.
