# BlindNav Care: Web Dashboard cho Người thân

Ứng dụng web giám sát thời gian thực dành cho người thân của người khiếm thị trong hệ thống điều hướng PBL6. Giao diện hỗ trợ theo dõi camera trực tuyến, đàm thoại hai chiều kiểu nhấn giữ nói qua loa thiết bị, bản đồ GPS luôn khóa tâm theo vị trí người dùng và trang cấu hình thông số toàn hệ thống.

## Cấu trúc thư mục

- [backend/app.py](backend/app.py): Máy chủ FastAPI cung cấp REST API, luồng video MJPEG và các kênh WebSocket cho telemetry, đàm thoại Push-to-Talk và cầu nối âm thanh thiết bị biên.
- [backend/cfg.py](backend/cfg.py): Quản lý đọc, kiểm tra hợp lệ và lưu trữ [config/dashboard_config.yaml](config/dashboard_config.yaml), hỗ trợ đồng bộ cấu hình sang [../2_edge_device/configs/config.yaml](../2_edge_device/configs/config.yaml) và [../2_edge_device/deployment_config.yaml](../2_edge_device/deployment_config.yaml).
- [backend/sim.py](backend/sim.py): Bộ mô phỏng luồng camera góc nhìn thứ nhất, vật cản 25 lớp ROD, cảm biến siêu âm HC-SR04 và quỹ đạo GPS để kiểm thử đầy đủ khi chưa kết nối Raspberry Pi thật.
- [backend/edge.py](backend/edge.py): Cầu nối nhận dữ liệu từ Raspberry Pi qua HTTP push, kéo trực tiếp từ dịch vụ cổng 8080 của `2_edge_device` hoặc qua MQTT, đồng thời chuyển tiếp âm thanh người thân xuống loa thiết bị qua WebSocket.
- [frontend/index.html](frontend/index.html), [frontend/styles.css](frontend/styles.css), [frontend/app.js](frontend/app.js): Giao diện 3 trang gồm `Trực tiếp & Đàm thoại`, `Bản đồ Thời gian thực` và `Cấu hình & Thông số`.
- [tests/test_web.py](tests/test_web.py): Bộ kiểm thử API, luồng ảnh JPEG, cấu hình YAML, WebSocket telemetry và WebSocket Push-to-Talk.

## Cách chạy nhanh trên máy cá nhân

Từ thư mục `3_web_dashboard`:

```bash
pip install -r requirements.txt
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8088
```

Sau đó mở trình duyệt tại `http://localhost:8088`.

## Chạy kiểm thử tự động

```bash
cd 3_web_dashboard
pytest tests/test_web.py -v
```

## Chạy bằng Docker Compose

```bash
cd 3_web_dashboard
docker compose up --build
```

## Kết nối với thiết bị biên Raspberry Pi 4

1. Vào trang **Cấu hình & Thông số**, chuyển **Nguồn dữ liệu** sang `Thiết bị Raspberry Pi thật`.
2. Nhập địa chỉ IP của Raspberry Pi và cổng HTTP mặc định `8080` của `2_edge_device`.
3. Thiết bị biên có thể đẩy telemetry JSON về `POST /api/edge/telemetry`, đẩy khung hình JPEG về `POST /api/edge/frame` và kết nối WebSocket `/ws/edge_audio` để nhận luồng PCM16 16kHz hoặc lệnh đọc TTS phát ra loa.
