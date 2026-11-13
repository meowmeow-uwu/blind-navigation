# BỘ CÔNG CỤ THỰC NGHIỆM ĐỐI SÁNH OBJECT TRACKING (TRACKING BENCHMARK SUITE)

> [!IMPORTANT]
> **Phân hệ độc lập (Standalone Benchmark Subsystem):** 
> Thư mục này chứa toàn bộ mã nguồn phục vụ việc thu thập dữ liệu video hiện trường và chạy thực nghiệm đối sánh 4 cấu hình bám vết đối tượng (`Raw YOLO`, `SORT`, `DeepSORT`, `Edge ByteTrack`).
> Bộ công cụ này **hoàn toàn độc lập** với mã nguồn hệ thống thời gian thực `main.py` trên Raspberry Pi 4.

---

## 1. Cấu Trúc Thành Phần

```
2_edge_device/experiments/tracking_benchmark/
├── record_scenarios.py             # 1. Quay video thực địa 12 kịch bản (thuần Camera, không siêu âm)
├── gt_annotator.py                 # 2. Gán nhãn Ground-Truth bán tự động (hỗ trợ phím 1..9, Space, D)
├── trackers.py                     # 3. Cài đặt 4 cấu hình tracker với cùng siêu tham số công bằng
├── run_benchmark.py                # 4. Replay tự động 4 tracker trên cùng tập video, xuất log CSV 15 cột
├── evaluate_results.py             # 5. Đọc log CSV & Ground-Truth -> tính MOTA, IDF1, IDSW, FPS, Nói lặp
├── test_tracking_benchmark_pipeline.py # 6. Tự động kiểm thử End-to-End toàn bộ pipeline
├── data/
│   ├── videos/                     # Nơi lưu 36 file video .mp4 (12 kịch bản x 3 runs)
│   └── ground_truth/               # Nơi lưu file Ground-Truth *_gt.csv
└── results/
    ├── logs/                       # Nơi lưu tệp log CSV <scenario>__<tracker>.csv
    └── tracking_benchmark_report.md # Báo cáo kết quả định lượng tự động
```

---

## 2. Quy Trình Vận Hành Thực Nghiệm (4 Bước)

### Bước 1: Thu thập video hiện trường (Đội phần cứng)
```bash
# Quay tự động lần lượt 12 kịch bản (mỗi kịch bản 3 runs, 25 giây/run):
python3 record_scenarios.py --all --runs 3 --duration 25 --source auto --countdown 5 --headless
```

### Bước 2: Gán nhãn Ground-Truth (Đội xử lý)
```bash
# Gán nhãn cho một kịch bản:
python3 gt_annotator.py --video data/videos/scen1_static_run1.mp4 --out-gt data/ground_truth/scen1_static_run1_gt.csv

# Tạo Ground-Truth rỗng cho Kịch bản 3 (hành lang trống không vật cản):
python3 gt_annotator.py --video data/videos/scen3_empty_run1.mp4 --out-gt data/ground_truth/scen3_empty_run1_gt.csv --empty-gt
```

### Bước 3: Chạy Replay đối sánh 4 Tracker
```bash
# Chạy cả 4 tracker trên tập video, xuất log CSV ra thư mục results/logs/:
python3 run_benchmark.py --videos-dir data/videos --out-dir results/logs --model yolov8n
```

### Bước 4: Tính toán Metrics & Xuất Báo Cáo
```bash
# Đối chiếu log với Ground-Truth, in bảng và xuất file tracking_benchmark_report.md:
python3 evaluate_results.py --logs-dir results/logs --gt-dir data/ground_truth --out-md results/tracking_benchmark_report.md
```

---

## 3. Kiểm Thử Hệ Thống Thực Nghiệm
Để kiểm tra tính toàn vẹn của toàn bộ pipeline mà không cần video thật:
```bash
python3 test_tracking_benchmark_pipeline.py
```
*(Tự động sinh dữ liệu kiểm thử, chạy 4 tracker, kiểm tra 15 cột log CSV và xác nhận các chỉ số MOTA/IDF1).*

