# PHÂN HỆ TÀI LIỆU THỰC NGHIỆM & THU THẬP DỮ LIỆU (BENCHMARK SUBSYSTEM)

> [!NOTE]
> Thư mục này chứa toàn bộ tài liệu hướng dẫn thực địa, phương pháp luận thực nghiệm và kế hoạch đối sánh khoa học của đề tài PBL6.
> Phân hệ này được **tách biệt hoàn toàn** khỏi mã nguồn và tài liệu hệ thống thời gian thực (`main.py`, `core/`, `modules/`).

---

## Danh Mục Tài Liệu Trong Phân Hệ

1. **[`huong_dan_thu_thap_du_lieu_thuc_dia.md`](huong_dan_thu_thap_du_lieu_thuc_dia.md):**
   - **Mục đích:** Quy trình Thao tác Chuẩn (SOP) dành cho Đội Thu thập Dữ liệu Hiện trường.
   - **Nội dung:** Chi tiết 12 kịch bản thực nghiệm video (đánh số tuần tự 1 $\rightarrow$ 12 qua 4 nhóm thử thách), chuẩn thời lượng 25 giây @ 15 FPS, tối thiểu 3 runs/kịch bản. 100% tập trung vào Camera Object Tracking (không dùng siêu âm).

2. **[`ke_hoach_thuc_nghiem_object_tracking.md`](ke_hoach_thuc_nghiem_object_tracking.md):**
   - **Mục đích:** Đặc tả phương pháp luận khoa học đối sánh 4 cấu hình: `Raw YOLO`, `SORT`, `DeepSORT`, và `Edge ByteTrack`.
   - **Nội dung:** Công thức toán học tính MOTA, IDF1, IDSW, Tỷ lệ nói lặp, Tỷ lệ cảnh báo nhầm, và nguyên tắc Record-Once Replay-All.

3. **[`ke_hoach_toan_dien_tracking_fusion_phan_cung.md`](ke_hoach_toan_dien_tracking_fusion_phan_cung.md):**
   - **Mục đích:** Kế hoạch giải quyết toàn diện 6 vấn đề kỹ thuật lớn (Rung lắc camera, thừa kế ByteTrack, phần cứng âm thanh loa rời, kiểm thử đối sánh, kịch bản Model Fusion, và SOP thực địa).

4. **[`ke_hoach_thuc_nghiem_va_bao_ve_t4.md`](ke_hoach_thuc_nghiem_va_bao_ve_t4.md):**
   - **Mục đích:** Kịch bản thực nghiệm tổng thể và lộ trình bảo vệ tiến độ trước Hội đồng.

---

## Mã Nguồn Đi Kèm

Toàn bộ bộ công cụ phần mềm thực thi thực nghiệm nằm tại:
`2_edge_device/experiments/tracking_benchmark/`
- `record_scenarios.py`: Công cụ quay 12 kịch bản video thực địa.
- `gt_annotator.py`: Công cụ gán nhãn Ground-Truth bán tự động.
- `trackers.py`: Cài đặt đồng nhất 4 cấu hình tracker.
- `run_benchmark.py`: Bộ thực thi replay đối sánh tự động.
- `evaluate_results.py`: Bộ tính toán chỉ số MOTA/IDF1 và xuất báo cáo Markdown.
- `test_tracking_benchmark_pipeline.py`: Kiểm thử tự động toàn bộ luồng thực nghiệm.
