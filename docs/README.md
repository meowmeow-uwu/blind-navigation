# TÀI LIỆU HỆ THỐNG HỖ TRỢ NGƯỜI KHIẾM THỊ THÔNG MINH (PBL6)

Hệ thống tài liệu dự án được cấu trúc thành **2 phân hệ hoàn toàn tách biệt**, tránh nhầm lẫn giữa hệ thống thời gian thực đang chạy và các công cụ thực nghiệm thu thập dữ liệu:

---

## 🏛️ PHÂN HỆ 1: HỆ THỐNG THỜI GIAN THỰC (PRODUCTION REAL-TIME SYSTEM)
*Tài liệu kỹ thuật đặc tả hệ thống đang chạy trực tiếp trên thiết bị đeo của người khiếm thị.*

| Tệp tài liệu | Nội dung & Mục tiêu |
| :--- | :--- |
| [`bao_cao_chuyen_sau_tru_cot_2_3_4.md`](bao_cao_chuyen_sau_tru_cot_2_3_4.md) | Báo cáo nghiên cứu học thuật chuẩn IEEE: Hợp nhất cảm biến (Pillar 2), ByteTrack (Pillar 3), Đánh giá nguy cơ (Pillar 4), Phản hồi âm thanh (Pillar 5). |
| [`bao_cao_chuyen_sau_hop_nhat_cam_bien.md`](bao_cao_chuyen_sau_hop_nhat_cam_bien.md) | Chuyên đề sâu về Hợp nhất cảm biến: Hình học mặt sàn, chùm sóng nón $30^\circ$, bộ lọc Kalman DWNA, và hiệu chỉnh thang đo trực tuyến. |
| [`huong_dan_lap_dat_phan_cung.md`](huong_dan_lap_dat_phan_cung.md) | Sơ đồ chân cắm GPIO, lắp ráp cụm camera, cảm biến siêu âm HC-SR04 và cấu hình loa. |
| [`co_so_khoa_hoc_va_nguon_goc_giai_phap.md`](co_so_khoa_hoc_va_nguon_goc_giai_phap.md) | Nguồn gốc các mô hình toán học và bài báo khoa học nền tảng. |
| [`byte_tracker_architecture_flowchart.svg`](byte_tracker_architecture_flowchart.svg) | Sơ đồ luồng xử lý 2 tầng IoU của thuật toán ByteTrack trên thiết bị biên. |
| [`sensor_fusion_architecture_flowchart.svg`](sensor_fusion_architecture_flowchart.svg) | Sơ đồ luồng hợp nhất dữ liệu Camera và Siêu âm theo thời gian thực. |

---

## 🔬 PHÂN HỆ 2: THỰC NGHIỆM & THU THẬP DỮ LIỆU (BENCHMARK SUBSYSTEM)
*Thư mục chuyên biệt: [`experiments_benchmark/`](experiments_benchmark/)*  
*Chứa toàn bộ hướng dẫn hiện trường, kịch bản quay video và phương pháp luận đối sánh khoa học.*

| Tệp tài liệu | Nội dung & Mục tiêu |
| :--- | :--- |
| [`experiments_benchmark/huong_dan_thu_thap_du_lieu_thuc_dia.md`](experiments_benchmark/huong_dan_thu_thap_du_lieu_thuc_dia.md) | **Cẩm nang hiện trường (SOP):** Hướng dẫn đội phần cứng quay 12 kịch bản video chuẩn (đánh số liên tục từ 1 đến 12), chuẩn 25s @ 15 FPS, tối thiểu 3 runs. 100% thuần Camera, không dùng siêu âm. |
| [`experiments_benchmark/ke_hoach_thuc_nghiem_object_tracking.md`](experiments_benchmark/ke_hoach_thuc_nghiem_object_tracking.md) | **Phương pháp luận Benchmark:** Thiết kế đối sánh 4 cấu hình (`Raw YOLO`, `SORT`, `DeepSORT`, `Edge ByteTrack`) theo nguyên tắc Record-Once Replay-All. |
| [`experiments_benchmark/ke_hoach_toan_dien_tracking_fusion_phan_cung.md`](experiments_benchmark/ke_hoach_toan_dien_tracking_fusion_phan_cung.md) | **Kế hoạch định hướng toàn diện:** Giải quyết camera rung lắc, định vị phần cứng âm thanh loa rời, kiểm thử đối sánh mô hình. |
| [`experiments_benchmark/ke_hoach_thuc_nghiem_va_bao_ve_t4.md`](experiments_benchmark/ke_hoach_thuc_nghiem_va_bao_ve_t4.md) | **Kế hoạch bảo vệ tiến độ:** Lộ trình thực nghiệm và các mốc báo cáo trước Hội đồng. |

---

## 🚀 MÃ NGUỒN TƯƠNG ỨNG

- **Hệ thống thời gian thực:** Nằm tại `2_edge_device/main.py`, `core/`, `modules/`, `configs/`.
- **Bộ công cụ thực nghiệm:** Nằm độc lập tại `2_edge_device/experiments/tracking_benchmark/` (xem [`README`](../2_edge_device/experiments/tracking_benchmark/README.md)).

