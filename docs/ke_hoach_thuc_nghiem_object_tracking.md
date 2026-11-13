# KẾ HOẠCH & QUY TRÌNH THỰC NGHIỆM KHÁCH QUAN CHO OBJECT TRACKING (PBL6)

Tài liệu này phản biện thiết kế thực nghiệm Object Tracking mà bạn đề xuất, chuẩn hóa phương pháp luận khoa học (Ablation Study & Benchmark công bằng trên thiết bị biên), đồng thời hướng dẫn từng bước vận hành bộ mã nguồn thực nghiệm đã xây dựng tại [`2_edge_device/experiments/tracking_benchmark/`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/) để thu thập số liệu thực tế trên Raspberry Pi 4.

> [!IMPORTANT]
> **Nguyên tắc trung thực khoa học:** Mọi bảng số liệu giả lập trước đây tuyệt đối **không** được đưa vào báo cáo cuối cùng. Toàn bộ số liệu `MOTA`, `IDF1`, `IDSW`, `FPS trên Pi 4`, `Tỷ lệ nói lặp` và `Cảnh báo nhầm` phải được sinh tự động từ tệp log thực tế do thiết bị ghi nhận thông qua [`run_benchmark.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/run_benchmark.py) và [`evaluate_results.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/evaluate_results.py).

---

## 1. Đánh giá & Phản biện Thiết kế Thực nghiệm Đề xuất

### 1.1. Thiết kế bạn đề xuất có đạt chuẩn học thuật không?
**Câu trả lời là CÓ.** Việc kiểm chứng cùng lúc 4 cấu hình (`Raw YOLO` → `SORT` → `DeepSORT` → `Edge ByteTrack`) qua 6 kịch bản biên và gắn chặt thước đo MOT chuẩn (`MOTA`, `IDF1`, `IDSW`) với thước đo trải nghiệm người khiếm thị (`Tỷ lệ nói lặp`, `Cảnh báo nhầm`) tạo thành một nghiên cứu cắt lớp (**Ablation + Comparative Study**) thuyết phục.

Tuy nhiên, để một nhà khoa học hoặc Hội đồng phản biện **không thể bắt bẻ tính thiên lệch thực nghiệm**, thiết kế thô ban đầu cần được khóa chặt **4 lỗ hổng phương pháp luận** dưới đây (tất cả đều đã được giải quyết trong bộ mã nguồn mới xây dựng):

| Lỗ hổng nếu làm thủ công | Rủi ro bị Hội đồng bác bỏ | Giải pháp khóa chặt đã cài đặt trong mã nguồn |
| :--- | :--- | :--- |
| **1. Đi bộ 4 lần riêng cho 4 thuật toán** | Mỗi lần đi khác tốc độ bước, góc nghiêng camera, thời điểm người cắt ngang và ánh sáng $\Rightarrow$ so sánh 4 tracker trên **4 dữ liệu đầu vào khác nhau** là vô nghĩa. | Áp dụng quy trình **Quay 1 lần — Phát lại tất cả (Record-Once, Replay-All)**: [`record_scenarios.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/record_scenarios.py) lưu video MP4 + siêu âm đồng bộ theo từng `frame_id`; [`run_benchmark.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/run_benchmark.py) phát lại **cùng 1 tập video** qua cả 4 cấu hình. |
| **2. Tham số & quy tắc xuất track không đồng đều** | Nếu ByteTrack giữ `max_age=12, min_hits=2` và xuất track dự báo trong khi SORT cài tham số khác, chênh lệch `MOTA`/`IDSW` bị nhiễu bởi siêu tham số thay vì thuật toán. | Khóa cứng cùng bộ tham số (`max_age=12`, `min_hits=2`, `iou_threshold=0.25`, `conf_thresh=0.45`) và **cùng quy tắc xuất `[t for t in tracks if t.hits >= min_hits]`** cho SORT, DeepSORT, ByteTrack trong [`trackers.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/trackers.py). Nhờ đó **100% khác biệt giữa SORT và Edge ByteTrack chỉ đến từ tầng cứu vớt `D_low` (`0.20–0.45`)**. |
| **3. Dùng đồng hồ hệ thống `time.time()` khi Replay** | Khi chạy offline (Mac nhanh ~49 FPS, DeepSORT trên Pi 4 chậm ~1.8 FPS), `time.time()` làm sai lệch hoàn toàn thời gian `cooldown = 2.5s` và tính sai `lần/phút`. | [`BenchmarkAudioSimulator`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/run_benchmark.py#L41-L119) sử dụng **đồng hồ thời gian của video (`video_time_s = frame_id / video_fps`)**, tái lập chính xác logic `main.py` bất kể tốc độ phần cứng phát lại. |
| **4. Mập mờ thành phần Re-ID của DeepSORT** | Gắn nhãn `"Có (CNN 128-D)"` khi chưa nạp mạng CNN thật sẽ vi phạm liêm chính học thuật; tính `MOTA/IDF1` trên kịch bản hành lang trống (`GT = 0`) bị lỗi chia 0. | [`ReIDFeatureExtractor`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/trackers.py#L137-L219) tự kiểm tra tệp `.onnx`. Có tệp CNN thật sẽ ghi rõ tên mô hình; chưa có sẽ chạy fallback `Sobel+HSV` và ghi nhãn tường minh `Sobel+HSV_Fallback (Chưa nạp ONNX CNN)` vào CSV & bảng. Riêng Kịch bản 6 hiển thị `—` cho MOTA/IDF1 và tách riêng chỉ số **Cảnh báo nhầm (lần/phút)**. |

---

## 2. Thiết kế Hệ thống 12 Kịch bản & Quy chuẩn Thu thập Dữ liệu (Đánh số 1 -> 12)

Toàn bộ video cần quay ở độ phân giải chuẩn **`640x480`, `15 FPS`**, gắn camera ở ngực (~`1.42m`, góc chúc ~`5°`). Thời lượng quy chuẩn chính thức: **25 giây / video** (~375 frames) và quay **lặp lại tối thiểu 3 runs** (`run 1, 2, 3`) cho mỗi kịch bản.

Chi tiết 12 kịch bản được đánh số tuần tự theo 4 nhóm thử thách (xem chi tiết tại [`huong_dan_thu_thap_du_lieu_thuc_dia.md`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/docs/huong_dan_thu_thap_du_lieu_thuc_dia.md)):
- **Nhóm I: Cơ bản & Đường đi đơn giản:**
  - `scen1_static`: Đi thẳng qua hành lang có 1 vật cản đứng yên (ghế 3.5m)
  - `scen2_multi`: Đi qua nhiều vật cản cách nhau so le (ghế trái 2m, thùng phải 3.5m)
  - `scen3_empty`: Hành lang trống hoàn toàn không có vật cản (đo False Alert Rate)
- **Nhóm II: Che khuất & Động lực học đối tượng:**
  - `scen4_occlusion`: Người đi ngang che khuất vật thể tĩnh trong 1.0–1.5s (ByteTrack cứu vết ở tầng 2)
  - `scen5_cross_paths`: 2 người đi bộ ngược chiều đan chéo cắt mặt (kiểm tra hoán đổi ID giữa 2 person)
  - `scen6_reenter`: Đối tượng rời khung hình rồi xuất hiện lại sau 1-2s (kiểm tra xóa track và khởi tạo ID mới)
- **Nhóm III: Chuyển động người đeo & Rung lắc camera (Ego-motion):**
  - `scen7_edge_to_center`: Vật thể từ rìa ảnh vào vùng chính giữa
  - `scen8_shake`: Camera rung lắc khi người đeo bước đi nhanh, dằn gót chân
  - `scen9_fast_approach`: Vật cản tiếp cận nhanh về phía camera (~1.5 m/s)
  - `scen10_turn_corner`: Cua góc hành lang 90 độ, vật cản bất ngờ xuất hiện
- **Nhóm IV: Môi trường khắc nghiệt & Mật độ dày đặc:**
  - `scen11_lighting_change`: Thay đổi ánh sáng đột ngột (tối sang sáng / ngược sáng)
  - `scen12_dense_clutter`: Môi trường đông đúc hỗn hợp nhiều đồ vật (ghế, thùng, người, balo)

---

## 3. Kiến trúc Bộ Mã nguồn Thực nghiệm & Lược đồ Log CSV

Tất cả mã nguồn nằm gọn trong thư mục [`2_edge_device/experiments/tracking_benchmark/`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/):

1. [`record_scenarios.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/record_scenarios.py): Quay video MP4 (`640x480`) kèm tệp `*_ultrasonic.csv` đồng bộ từng `frame_id` (`frame_id, timestamp, elapsed_s, us_distance_m, is_sensor_emergency`).
2. [`gt_annotator.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/gt_annotator.py): Công cụ gán nhãn Ground-Truth bán tự động (`frame_id, gt_id, x1, y1, x2, y2, class_name`), gợi ý sẵn khung bằng `YOLOv8n NCNN` + `LightweightTracker`, hỗ trợ click đổi `gt_id` liên hoàn cho mọi frame sau (`1..9`), xóa nhiễu (`D`/chuột phải), vẽ tay hộp bị che khuất (`D_low` hoặc mất hẳn) và `--empty-gt` cho `scen6_empty`.
3. [`trackers.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/trackers.py): Cài đặt đồng nhất 4 cấu hình từ lớp nền `BaseBenchmarkTracker`:
   - `raw_yolo`: [`RawNoTracker`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/trackers.py#L234-L255) (`conf >= 0.20`, gán ID mới mỗi frame).
   - `sort`: [`SORTTracker`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/trackers.py#L258-L318) (`KalmanBox2D` 7 trạng thái + Hungarian IoU chỉ trên `conf >= 0.45`).
   - `deepsort`: [`DeepSORTTracker`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/trackers.py#L321-L459) (`KalmanBox2D` + Matching Cascade Cosine trên vector 128-D + vòng fallback IoU trên `conf >= 0.45`; hỗ trợ nạp CNN `.onnx` thật qua `cv2.dnn.readNetFromONNX` hoặc fallback `Sobel+HSV` có gắn nhãn rõ).
   - `edge_bytetrack`: [`EdgeByteTrackWrapper`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/trackers.py#L462-L490) bọc trực tiếp [`LightweightTracker`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/modules/tracking/byte_tracker.py#L121-L276) của dự án.
4. [`run_benchmark.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/run_benchmark.py): Chạy tuần tự 4 cấu hình trên cùng tập video qua chuỗi `YoloNcnnRunner` (`target_size=320, conf_thresh=0.20`) $\rightarrow$ Tracker $\rightarrow$ `SensorFusionEngine` $\rightarrow$ `BenchmarkAudioSimulator`, xuất file CSV `<scenario>__<tracker>.csv` gồm đúng **15 cột**:
   - `timestamp`, `frame_id`, `scenario_id`, `tracker_name`, `reid_info`, `bbox`, `class_name`, `confidence`, `track_id`, `det_time_ms`, `track_time_ms`, `processing_time_ms`, `distance_estimate`, `audio_event`, `audio_track_id`.
5. [`evaluate_results.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/evaluate_results.py): Tự động đối chiếu toàn bộ tệp log `*__*.csv` với `*_gt.csv`, tính mọi thước đo học thuật & HCI, in bảng ra Terminal và xuất báo cáo `results/tracking_benchmark_report.md`.
6. [`test_tracking_benchmark_pipeline.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/test_tracking_benchmark_pipeline.py): Tự động kiểm thử toàn bộ luồng dữ liệu, schema 15 cột và tính đúng đắn của công thức đánh giá trên dữ liệu kiểm thử có kiểm soát.

---

## 4. Định nghĩa Toán học Chính xác Các Thước đo trong [`evaluate_results.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/evaluate_results.py)

1. **MOTA (%) — Multi-Object Tracking Accuracy** (Bernardin & Stiefelhagen, 2008):
   $$\text{MOTA} = \left(1 - \frac{\sum_t (\text{FN}_t + \text{FP}_t + \text{IDSW}_t)}{\sum_t \text{GT}_t}\right) \times 100\%$$
   Tại mỗi frame $t$, cặp `(GT, Hypothesis)` có $\text{IoU} \ge 0.50$ được ghép tối ưu 1–1 bằng thuật toán Hungarian (`linear_sum_assignment`). Một khung GT không ghép được tính là `FN`, một hypothesis dư tính là `FP`, và một `gt_id` bị đổi sang `track_id` khác với lần gán trước đó tính là `IDSW`.

2. **IDF1 (%) — Identification F1 Score** (Ristani et al., ECCVW 2016):
   $$\text{IDF1} = \frac{2 \cdot \text{IDTP}}{2 \cdot \text{IDTP} + \text{IDFP} + \text{IDFN}} \times 100\%$$
   Thực hiện ghép cặp toàn cục trên toàn bộ chuỗi thời gian giữa tập `gt_id` và tập `track_id` theo tổng số khung hình trùng khớp $\text{overlap}(g, \tau)$ để đo độ bền danh tính xuyên suốt video.

3. **FPS thực tế & Latency (`processing_time_ms`)**:
   $$\text{FPS} = \frac{1000.0}{\overline{T}_{\text{proc}}}, \quad \text{với } T_{\text{proc}} = T_{\text{det}} + T_{\text{track}} + T_{\text{fusion}} \text{ (ms/frame)}$$
   Tính trung bình có trọng số theo tổng số khung hình trên tất cả kịch bản.

4. **Tỷ lệ nói lặp (`lần/phút`)**:
   $$\text{Repeat Rate} = \frac{\sum_{g \in \text{GT}} \max(0,\, N_{\text{guidance}}(g) - 1)}{T_{\text{duration\_min}}}$$
   Đếm tổng số lần sự kiện âm thanh `guidance` phát **thừa từ lần thứ 2 trở đi** trên cùng một đối tượng Ground-Truth $g$ (dù do đứt ID hay tái thông báo), quy đổi về số lần/phút.

5. **Tỷ lệ cảnh báo nhầm (`lần/phút`, Kịch bản 6)**:
   $$\text{False Alert Rate} = \frac{N_{\text{audio\_events}(\text{GT}=0)}}{T_{\text{scen6\_min}}}$$

---

## 5. Quy trình Thực thi Từng Bước (SOP từ Hiện trường Pi 4 đến Báo cáo)

### Bước 1 — Kiểm tra nhanh bộ công cụ trên máy (trước khi ra hiện trường)
Tại thư mục `blind-navigation`:
```bash
cd /Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation
.venv/bin/python 2_edge_device/experiments/tracking_benchmark/test_tracking_benchmark_pipeline.py
```
Kiểm tra lệnh in ra `✅ TẤT CẢ KIỂM THỬ PIPELINE BENCHMARK OBJECT TRACKING ĐỀU ĐẠT!`.

### Bước 2 — (Tùy chọn) Chuẩn bị mô hình CNN Re-ID thật cho DeepSORT
Nếu nhóm muốn thử nghiệm **DeepSORT kèm mạng CNN 128-D thật** để đo chính xác độ sụt FPS trên CPU ARM Cortex-A72 của Raspberry Pi 4 (thay vì dùng bộ mô tả dự phòng `Sobel+HSV`), hãy đặt tệp trọng số ONNX (ví dụ `osnet_x0_25.onnx` hoặc `mars_small128.onnx`) vào:
```text
2_edge_device/modules/tracking/models/osnet_x0_25.onnx
```
*(hoặc truyền đường dẫn trực tiếp qua cờ `--onnx-reid /duong/dan/model.onnx` ở Bước 5).*
Nếu chưa có tệp `.onnx`, **vẫn chạy bình thường** — mã nguồn sẽ dùng fallback `Sobel+HSV` và tự động ghi rõ `Sobel+HSV_Fallback (Chưa nạp ONNX CNN)` vào bảng để đảm bảo trung thực với Hội đồng.

### Bước 3 — Quay 1 lần 6 video kịch bản chuẩn và lưu về máy
[`record_scenarios.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/record_scenarios.py) tự động lưu trực tiếp tệp `.mp4` (`640x480 @ 15 FPS`) cùng log siêu âm `*_ultrasonic.csv` vào `2_edge_device/experiments/tracking_benchmark/data/videos/`. Tùy hoàn cảnh thiết bị, có **3 cách thực hiện**:

- **Cách A (Chuẩn nhất — Quay ngoài hành lang bằng Raspberry Pi 4 rồi chuyển về Mac):**
  1. Cấp nguồn Raspberry Pi 4 bằng pin/sạc dự phòng, bật Hotspot điện thoại để Mac/điện thoại SSH vào Pi 4.
  2. Chạy quay từng kịch bản (nên quay lẻ `--scenario ... --headless --countdown 5` để Pi tự đếm ngược 5 giây cho bạn vào vị trí đi bộ, quay xong 15 giây sẽ tự động dừng và lưu file):
     ```bash
     cd ~/blind-navigation/2_edge_device/experiments/tracking_benchmark
     python3 record_scenarios.py --scenario scen1_static --duration 15 --fps 15 --source auto --countdown 5 --headless
     # Lặp lại cho: scen2_multi, scen3_occlusion, scen4_edge_to_center, scen5_shake, scen6_empty
     ```
  3. **Lấy thư mục video từ Pi 4 về Mac (chọn 1 trong 3 cách):**
     - *Terminal trên Mac (`scp`):*
       ```bash
       scp -r <user_pi>@<ip_pi>:~/blind-navigation/2_edge_device/experiments/tracking_benchmark/data/videos \
         /Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/data/
       ```
     - *Qua trình duyệt (không cần gõ `scp`):* Trên Pi 4 mở server tạm trong thư mục `data/videos/`:
       ```bash
       cd ~/blind-navigation/2_edge_device/experiments/tracking_benchmark/data/videos && python3 -m http.server 8000
       ```
       Sau đó mở Safari/Chrome trên Mac vào `http://<ip_pi>:8000` để bấm tải từng file `.mp4` và `.csv` về thư mục `data/videos/` trên Mac.
     - *VS Code Remote-SSH / VNC / USB:* Chuột phải vào thư mục `data/videos/` trong VS Code chọn *Download...* hoặc chép qua USB.

- **Cách B (Tiện nhất nếu test nhanh — Cắm USB Webcam trực tiếp vào MacBook):**
  Chạy ngay trên máy Mac thì file `.mp4` lưu thẳng vào `data/videos/` trên Mac (lưu ý: sẽ không có dữ liệu GPIO siêu âm HC-SR04 thật, CSV ghi mặc định `2.2m`; sau khi gán nhãn cần chép `data/videos/` sang Pi 4 để đo FPS thật ở Bước 5):
  ```bash
  .venv/bin/python 2_edge_device/experiments/tracking_benchmark/record_scenarios.py \
    --scenario scen1_static --duration 15 --fps 15 --source auto --countdown 5
  ```

- **Cách C (Chuẩn hóa từ video quay sẵn bằng điện thoại/máy ảnh):**
  Truyền đường dẫn file `.mp4`/`.mov` vào `--source` để tự động resize về `640x480` và resample đúng `15 FPS` vào `data/videos/` *(chỉ dùng tham khảo khi chưa gắn được webcam chuẩn vì góc nhìn/tiêu cự điện thoại khác webcam)*:
  ```bash
  .venv/bin/python 2_edge_device/experiments/tracking_benchmark/record_scenarios.py \
    --scenario scen1_static --source /duong/dan/video_goc.mp4 --fps 15 --headless
  ```

### Bước 4 — Gán nhãn Ground-Truth bán tự động trên Mac/PC
Sau khi các tệp `.mp4` đã nằm trong `2_edge_device/experiments/tracking_benchmark/data/videos/` trên Mac:
```bash
# Kịch bản 1..5: mở cửa sổ gán nhãn bán tự động
.venv/bin/python 2_edge_device/experiments/tracking_benchmark/gt_annotator.py \
  --video 2_edge_device/experiments/tracking_benchmark/data/videos/scen1_static.mp4 \
  --out-gt 2_edge_device/experiments/tracking_benchmark/data/ground_truth/scen1_static_gt.csv
# (Lặp lại tương tự cho scen2_multi, scen3_occlusion, scen4_edge_to_center, scen5_shake)

# Kịch bản 6 (Hành lang trống): tạo nhanh Ground-Truth rỗng
.venv/bin/python 2_edge_device/experiments/tracking_benchmark/gt_annotator.py \
  --video 2_edge_device/experiments/tracking_benchmark/data/videos/scen6_empty.mp4 \
  --out-gt 2_edge_device/experiments/tracking_benchmark/data/ground_truth/scen6_empty_gt.csv \
  --empty-gt
```
**Mẹo thao tác nhanh trong `gt_annotator.py`:** Chú ý đặc biệt ở đoạn che khuất (`scen3_occlusion`) và rung lắc (`scen5_shake`), nếu YOLO gợi ý bị mất hộp hoặc bị nhảy `gt_id`, click vào hộp rồi nhấn phím `[1..9]` để thống nhất lại ID cho toàn bộ chuỗi về sau, hoặc kéo chuột trái để vẽ hộp bổ sung cho vật bị che khuất một phần.

### Bước 5 — Chạy Replay Benchmark trực tiếp trên Raspberry Pi 4
> [!WARNING]
> **Bắt buộc chạy bước này ngay trên Raspberry Pi 4 thật** (không dùng `--cache-dets`) nếu muốn lấy cột **`FPS trên Pi 4`** và **`Latency (ms/frame)`** phản ánh đúng năng lực phần cứng biên, nhiệt độ và tải CPU ARM thật. (Cờ `--cache-dets` chỉ dùng khi phát lại trên Mac để kiểm tra nhanh `MOTA/IDF1/IDSW`).

```bash
python3 run_benchmark.py \
  --videos-dir data/videos \
  --out-dir results/logs \
  --model yolov8n \
  --trackers raw_yolo sort deepsort edge_bytetrack
# Nếu đã chuẩn bị file Re-ID ONNX cho DeepSORT, thêm: --onnx-reid ../../modules/tracking/models/osnet_x0_25.onnx
```
Sau bước này, `results/logs/` sẽ chứa `24 tệp log CSV` (6 kịch bản $\times$ 4 cấu hình tracker).

### Bước 6 — Tính Metrics và Xuất Bảng Báo cáo Tự động
```bash
python3 evaluate_results.py \
  --logs-dir results/logs \
  --gt-dir data/ground_truth \
  --out-md results/tracking_benchmark_report.md
```
Lệnh sẽ in bảng đối sánh tổng hợp ngay trên màn hình Terminal và lưu báo cáo chi tiết từng kịch bản vào `results/tracking_benchmark_report.md`.

---

## 6. Kết quả Tự kiểm thử Logic Phần mềm (Synthetic Sanity Check) & Khung Bảng Thực địa

### 6.1. Kết quả kiểm thử End-to-End đã chạy thành công (Không phải số đo Pi 4)
Trước khi triển khai ra ngoài hành lang trên Raspberry Pi 4, bộ mã nguồn đã vượt qua bài tự kiểm thử có kiểm soát [`test_tracking_benchmark_pipeline.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/test_tracking_benchmark_pipeline.py) trên máy Mac (sử dụng 4 kịch bản tổng hợp dài 90 frame @ 15 FPS, thời gian detection giả lập cố định `20.0 ms`, `scen3_occlusion` giả lập vật thể di chuyển bị che khuất 15 frame ở mức `conf=0.30`, `scen6_empty` chèn 3 xung nhiễu giả 1-frame tại khoảng ~2.6m):

| Thuật toán Tracker | Thành phần Re-ID ghi nhận | MOTA (%) | IDF1 (%) | IDSW | Repeat (lần/phút) | False Alerts ở Kịch bản trống (lần/phút) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `raw_yolo` | Không | 0.0% | 1.1% | 267 | 15.0 | 5.0 |
| `sort` | Không | 95.2% | 71.5% | 2 | 5.0 | 0.0 |
| `deepsort` | `Sobel+HSV_Fallback (Chưa nạp ONNX CNN)` | 95.2% | 71.5% | 2 | 5.0 | 0.0 |
| `edge_bytetrack` | `Không (2-Stage IoU)` | **98.9%** | **99.4%** | **0** | **0.0** | **0.0** |

> [!NOTE]
> Bảng 6.1 chứng minh **mã nguồn tính toán metrics, tách tầng `D_high/D_low`, bộ lọc `min_hits=2` và logic chống nói lặp hoạt động chính xác**. Trên video tổng hợp có pha `D_low` kéo dài 15 frames (> `max_age=12`), `SORT` và `DeepSORT` bỏ qua detection mờ nên đứt vết (`IDSW=2`, `Repeat=5.0 lần/phút`), trong khi `Edge ByteTrack` giữ nguyên `IDSW=0` và `Repeat=0.0 lần/phút`. Tuy nhiên đây **chưa phải kết quả đo đạc trên phần cứng Raspberry Pi 4 và video thực tế**.

### 6.2. Khung Bảng Kết quả Thực nghiệm Thực tế (Điền từ `results/tracking_benchmark_report.md` sau Bước 6)

| Thuật toán Tracker | Thành phần Re-ID (Thật/Fallback) | MOTA (%) | IDF1 (%) | ID Switches (IDSW) | FPS trên Pi 4 | Track Latency (ms) | Tỷ lệ nói lặp (lần/phút) | Cảnh báo nhầm KB6 (lần/phút) | Đánh giá khoa học trên thiết bị biên |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw YOLO (Không tracker)** | Không | *(Điền từ log)* | *(Điền từ log)* | *(Điền)* | *(Pi 4)* | *(ms)* | *(lần/p)* | *(lần/p)* | *(Nhận xét từ dữ liệu)* |
| **SORT (Bewley et al., 2016)** | Không | *(Điền từ log)* | *(Điền từ log)* | *(Điền)* | *(Pi 4)* | *(ms)* | *(lần/p)* | *(lần/p)* | *(Nhận xét từ dữ liệu)* |
| **DeepSORT (Wojke et al., 2017)** | *(Ghi đúng nhãn log)* | *(Điền từ log)* | *(Điền từ log)* | *(Điền)* | *(Pi 4)* | *(ms)* | *(lần/p)* | *(lần/p)* | *(Nhận xét từ dữ liệu)* |
| **Edge ByteTrack (PBL6)** | Không (2-Stage IoU) | *(Điền từ log)* | *(Điền từ log)* | *(Điền)* | *(Pi 4)* | *(ms)* | *(lần/p)* | *(lần/p)* | *(Nhận xét từ dữ liệu)* |
