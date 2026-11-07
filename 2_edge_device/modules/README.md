# 🧩 Edge Device Core Modules Reference

Thư mục `modules/` chứa các thành phần kiến trúc độc lập (micro-modules) phục vụ cho hệ thống điều hướng trên thiết bị biên. Mỗi module tuân thủ nguyên tắc trách nhiệm duy nhất (Single Responsibility Principle) và không phụ thuộc vòng lặp (no circular dependencies).

---

## 📁 Sơ đồ các Sub-Modules

| Thư mục | Trách nhiệm chính | Lớp / File cốt lõi | Đầu vào / Đầu ra chính |
| :--- | :--- | :--- | :--- |
| **[`vision_local/`](./vision_local)** | Nhận diện vật thể cục bộ & bám vết đa đối tượng | `YoloOnnxRunner`<br>`YoloNcnnRunner`<br>`ByteTrackerWrapper` | **In:** BGR Frame ($640 \times 480$)<br>**Out:** Tracked objects với Track ID, BBox, Confidence |
| **[`decision/`](./decision)** | Xếp hạng rủi ro va chạm & sinh chỉ dẫn né tránh | `PriorityScorer`<br>`TrackStateManager`<br>`NavigationPolicy` | **In:** Tracked objects<br>**Out:** Thông điệp cảnh báo tiếng Việt ("Né sang phải", "Dừng lại"...) |
| **[`audio/`](./audio)** | Phát thanh âm thanh độ trễ thấp tới người dùng | `AsyncAudioBroker` | **In:** Chuỗi văn bản tiếng Việt<br>**Out:** Âm thanh WAV (espeak-ng) & HTTP Stream (port 8080) |
| **[`vision_cloud/`](./vision_cloud)** | Trợ lý mô tả cảnh quan không gian qua mô hình VLM | `vision_cloud_runner.py` | **In:** Frame ảnh tĩnh (384px JPEG)<br>**Out:** Đoạn văn bản 1-2 câu miêu tả môi trường (Qwen-VL / LLaVA) |
| **[`utils/`](./utils)** | Bộ công cụ tiện ích phụ trợ đo đạc và trực quan hóa | `PipelineProfiler`<br>`StreamLoader`<br>`Visualizer` | Phân tích latency 5 stage, đồng bộ FPS video và vẽ HUD trực quan |

---

## 🔗 Liên kết mã nguồn
- Xem hướng dẫn chi tiết toàn bộ thiết bị biên tại: [**Module 2 README**](../README.md).
- Điểm khởi chạy chương trình: [**`main.py`**](../main.py) và [**`pipeline_runner.py`**](../pipeline_runner.py).
