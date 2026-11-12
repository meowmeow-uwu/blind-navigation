# Báo Cáo Audit Artifact & Evidence Độc Lập (Verification Audit)

> **Phạm vi kiểm toán:** Toàn bộ artifact ONNX, NCNN, INT8 ở profile 320×320 và 640×640 trong `blind-navigation/1_model_development/`.  
> **Nguyên tắc kiểm toán tối cao:** Chỉ đọc và đối chiếu file/báo cáo hiện có; **tuyệt đối KHÔNG chạy lại** inference, evaluation, export, quantization, conversion hoặc benchmark. **KHÔNG chỉnh sửa** `weights/manifest.json`, `README.md` hoặc gói triển khai `artifacts/pi_deployment_package.tar.gz`.

---

## 1. Tóm Tắt Kết Quả Kiểm Toán (Executive Summary)

Kiểm toán đã rà soát toàn bộ **24 artifact triển khai** (4 mô hình × 6 cấu hình: ONNX FP32 320, INT8 320, NCNN FP32 320, ONNX FP32 640, INT8 640, NCNN FP32 640) đối chiếu trực tiếp giữa [weights/manifest.json](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/weights/manifest.json) và các báo cáo thực nghiệm trong [reports/pipeline_run/](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/reports/pipeline_run/).

| Phân Loại (Audit Status) | Số Lượng | Tỷ Lệ | Mô Tả |
| :--- | :---: | :---: | :--- |
| **`verified`** | **8 / 24** | 33.3% | Có report chạy toàn bộ tập `rod_clean_v1/test` (1,615 ảnh thật, metric mAP50, mAP50–95, Recall, AP-small thật). Gồm ONNX FP32 640 (4 mô hình) và ONNX INT8 640 (4 mô hình). |
| **`proxy`** | **4 / 24** | 16.7% | Số liệu sao chép nguyên trạng từ ONNX FP32 sang NCNN FP32 640; không phải số liệu chạy trực tiếp từ `.param`/`.bin` trên tập test. |
| **`missing_evidence`** | **12 / 24** | 50.0% | Thiếu hoàn toàn báo cáo đánh giá độ chính xác (mAP) trên tập test thật. Gồm toàn bộ 12 artifact thuộc Profile 320 (ONNX, NCNN, INT8). |
| **`incorrect_flag`** | **16 / 24** | 66.7% | Cờ `accuracy_passed: true` trong `manifest.json` bị gán sai quy chuẩn: gán khi chỉ có smoke test/equivalence test (12 artifact 320) hoặc gán cho model proxy chưa chạy trực tiếp (4 artifact NCNN 640). |

---

## 2. Bảng Đối Chiếu Ma Trận Kiểm Toán 24 Artifact (Audit Matrix)

| Model | Profile | Format | Artifact File | Manifest `accuracy_passed` | Bằng Chứng Hiện Có (Report File & Loại Test) | Phân Loại Audit (`accuracy_status`) | Đánh Giá Cờ (`flag_assessment`) |
| :--- | :---: | :---: | :--- | :---: | :--- | :---: | :---: |
| **yolov8n** | 320 | ONNX FP32 | `yolov8n_320_fp32.onnx` | `true` | `onnx_equivalence_report.json` (Cosine sim trên 10 ảnh mẫu) | `missing_evidence` | `incorrect_flag` |
| **yolov8n** | 320 | INT8 PTQ | `yolov8n_320_int8.onnx` | `true` | `int8_ptq_evaluation_report.json` (Dummy tensor sanity check) | `missing_evidence` | `incorrect_flag` |
| **yolov8n** | 320 | NCNN FP32 | `yolov8n_ncnn_fp32` | `true` | `ncnn_conversion_summary.json` (Trạng thái convert, 0 test) | `missing_evidence` | `incorrect_flag` |
| **yolov8n** | 640 | ONNX FP32 | `yolov8n_rod_640_onnx_fp32.onnx` | `true` | `onnx_fp32_640_accuracy_report.json` (Full test 1,615 ảnh, mAP50=0.8990) | **`verified`** | **`verified`** |
| **yolov8n** | 640 | NCNN FP32 | `yolov8n_rod_640_ncnn_fp32` | `true` | `ncnn_fp32_640_accuracy_report.json` (Proxy copy từ ONNX FP32) | **`proxy`** | `incorrect_flag` |
| **yolov8n** | 640 | INT8 PTQ | `yolov8n_rod_640_onnx_int8.onnx` | `true` | `int8_ptq_640_accuracy_report.json` (Full test 1,615 ảnh, mAP50=0.8945) | **`verified`** | **`verified`** |
| **yolov9t** | 320 | ONNX FP32 | `yolov9t_320_fp32.onnx` | `true` | `onnx_equivalence_report.json` (Cosine sim trên 10 ảnh mẫu) | `missing_evidence` | `incorrect_flag` |
| **yolov9t** | 320 | INT8 PTQ | `yolov9t_320_int8.onnx` | `true` | `int8_ptq_evaluation_report.json` (Dummy tensor sanity check) | `missing_evidence` | `incorrect_flag` |
| **yolov9t** | 320 | NCNN FP32 | `yolov9t_ncnn_fp32` | `true` | `ncnn_conversion_summary.json` (Trạng thái convert, 0 test) | `missing_evidence` | `incorrect_flag` |
| **yolov9t** | 640 | ONNX FP32 | `yolov9t_rod_640_onnx_fp32.onnx` | `true` | `onnx_fp32_640_accuracy_report.json` (Full test 1,615 ảnh, mAP50=0.9134) | **`verified`** | **`verified`** |
| **yolov9t** | 640 | NCNN FP32 | `yolov9t_rod_640_ncnn_fp32` | `true` | `ncnn_fp32_640_accuracy_report.json` (Proxy copy từ ONNX FP32) | **`proxy`** | `incorrect_flag` |
| **yolov9t** | 640 | INT8 PTQ | `yolov9t_rod_640_onnx_int8.onnx` | `true` | `int8_ptq_640_accuracy_report.json` (Full test 1,615 ảnh, mAP50=0.9117) | **`verified`** | **`verified`** |
| **yolov12n** | 320 | ONNX FP32 | `yolov12n_320_fp32.onnx` | `true` | `onnx_equivalence_report.json` (Cosine sim trên 10 ảnh mẫu) | `missing_evidence` | `incorrect_flag` |
| **yolov12n** | 320 | INT8 PTQ | `yolov12n_320_int8.onnx` | `true` | `int8_ptq_evaluation_report.json` (Dummy tensor sanity check) | `missing_evidence` | `incorrect_flag` |
| **yolov12n** | 320 | NCNN FP32 | `yolov12n_ncnn_fp32` | `true` | `ncnn_conversion_summary.json` (Trạng thái convert, 0 test) | `missing_evidence` | `incorrect_flag` |
| **yolov12n** | 640 | ONNX FP32 | `yolov12n_rod_640_onnx_fp32.onnx` | `true` | `onnx_fp32_640_accuracy_report.json` (Full test 1,615 ảnh, mAP50=0.9071) | **`verified`** | **`verified`** |
| **yolov12n** | 640 | NCNN FP32 | `yolov12n_rod_640_ncnn_fp32` | `true` | `ncnn_fp32_640_accuracy_report.json` (Proxy copy từ ONNX FP32) | **`proxy`** | `incorrect_flag` |
| **yolov12n** | 640 | INT8 PTQ | `yolov12n_rod_640_onnx_int8.onnx` | `true` | `int8_ptq_640_accuracy_report.json` (Full test 1,615 ảnh, mAP50=0.9047) | **`verified`** | **`verified`** |
| **yolov26n** | 320 | ONNX FP32 | `yolov26n_320_fp32.onnx` | `true` | `onnx_equivalence_report.json` (Cosine sim trên 10 ảnh mẫu) | `missing_evidence` | `incorrect_flag` |
| **yolov26n** | 320 | INT8 PTQ | `yolov26n_320_int8.onnx` | `true` | `int8_ptq_evaluation_report.json` (Dummy tensor sanity check) | `missing_evidence` | `incorrect_flag` |
| **yolov26n** | 320 | NCNN FP32 | `yolov26n_ncnn_fp32` | `true` | `ncnn_conversion_summary.json` (Trạng thái convert, 0 test) | `missing_evidence` | `incorrect_flag` |
| **yolov26n** | 640 | ONNX FP32 | `yolov26n_rod_640_onnx_fp32.onnx` | `true` | `onnx_fp32_640_accuracy_report.json` (Full test 1,615 ảnh, mAP50=0.9064) | **`verified`** | **`verified`** |
| **yolov26n** | 640 | NCNN FP32 | `yolov26n_rod_640_ncnn_fp32` | `true` | `ncnn_fp32_640_accuracy_report.json` (Proxy copy từ ONNX FP32) | **`proxy`** | `incorrect_flag` |
| **yolov26n** | 640 | INT8 PTQ | `yolov26n_rod_640_onnx_int8.onnx` | `true` | `int8_ptq_640_accuracy_report.json` (Full test 1,615 ảnh, mAP50=0.9062) | **`verified`** | **`verified`** |

---

## 3. Chi Tiết Audit Theo 7 Trọng Tâm Yêu Cầu

### 3.1. Kiểm Tra Full-Test Thật vs Smoke / Proxy (320 vs 640)
* **Profile 640:**
  * `onnx_fp32_640`: **Report full-test thật**. File [reports/pipeline_run/onnx_fp32_640_accuracy_report.json](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/reports/pipeline_run/onnx_fp32_640_accuracy_report.json) lưu kết quả đánh giá trên toàn bộ 1,615 ảnh test của `rod_clean_v1`. Thời gian chạy thực tế từ 15.16s đến 19.45s mỗi model.
  * `onnx_int8_640`: **Report full-test thật**. File [reports/pipeline_run/int8_ptq_640_accuracy_report.json](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/reports/pipeline_run/int8_ptq_640_accuracy_report.json) ghi nhận mAP50, mAP50–95, per-class recall, AP-small đầy đủ sau 135s – 232s đánh giá CPU trên tập test.
  * `ncnn_fp32_640`: **Report proxy**. Không có inference trực tiếp từ engine NCNN.
* **Profile 320:**
  * Không có bất kỳ artifact nào được chạy full-test mAP trên `rod_clean_v1/test`.
  * `onnx_equivalence_report.json` chỉ chạy kiểm tra tương đương số học trên 10 ảnh (`IMG_19187.jpg` đến `IMG_19197.jpg`).
  * `int8_ptq_evaluation_report.json` chỉ chạy `dummy_input = torch.randn(1, 3, 320, 320)`.
  * `ncnn_conversion_summary.json` chỉ kiểm tra file sinh ra thành công hay không.

### 3.2. Đối Chiếu `accuracy_passed` Trong Manifest với Bằng Chứng
Quy tắc kiểm soát: *"Chỉ đặt accuracy_passed=true sau khi artifact đã chạy toàn bộ clean test; graph check, cosine similarity hoặc smoke test không được thay thế accuracy evaluation."*
* **Điểm không khớp nghiêm trọng (16/24 artifacts):**
  1. Cả 12 artifact ở profile 320 đều có `accuracy_passed: true` trong [weights/manifest.json](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/weights/manifest.json), dù bằng chứng lưu trữ chỉ là smoke test và cosine similarity 10 ảnh. Đây là **`incorrect_flag`**.
  2. Cả 4 artifact NCNN 640 có `accuracy_passed: true` trong `manifest.json`, dù chính file báo cáo thực nghiệm ghi rõ là ONNX FP32 proxy và trạng thái NCNN on-device là `Pending`. Đây là **`incorrect_flag`**.
* **Điểm khớp chuẩn xác (8/24 artifacts):**
  * 4 model ONNX FP32 640 và 4 model ONNX INT8 640 có `accuracy_passed: true` hoàn toàn trùng khớp với bằng chứng full clean test 1,615 ảnh.

### 3.3. NCNN: Trực Tiếp Từ `.param`/`.bin` Hay Sao Chép Từ ONNX (Proxy)
* **Kết luận:** NCNN 640 **hoàn toàn là số liệu sao chép (proxy)** từ ONNX FP32.
* **Bằng chứng từ mã nguồn và file JSON:**
  * File [reports/pipeline_run/ncnn_fp32_640_accuracy_report.json:L92](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/reports/pipeline_run/ncnn_fp32_640_accuracy_report.json#L92):
    ```json
    "accuracy_note": "Host accuracy evaluated via ONNX FP32 proxy (identical weights). True on-device NCNN inference accuracy requires Pi 4 hardware. Status: Pending."
    ```
  * Số liệu mAP50, mAP50–95, precision, recall, ap_small của 4 model NCNN trong file này giống 100% đến từng chữ số thập phân với [reports/pipeline_run/onnx_fp32_640_accuracy_report.json](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/reports/pipeline_run/onnx_fp32_640_accuracy_report.json).
  * Mã nguồn thực thi [scripts/run_640_profile.py:L360-L373](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/scripts/run_640_profile.py#L360-L373) cho thấy sau khi convert sang NCNN, pipeline nạp lại file `.onnx` để chạy test proxy chứ không chạy engine `ncnn.Net()`.
* **Ghi nhận trạng thái:** Theo đúng chỉ thị của người dùng, phân loại cho 4 artifact NCNN 640 là **`accuracy_status: "proxy"`**, giữ nguyên artifact model, không sửa đổi model.

### 3.4. Profile 320: Có Report Full-Test Riêng Hay Bị Gán Thủ Công?
* **Kết luận:** Profile 320 **bị gán `accuracy_passed: true` thủ công / từ smoke test**, hoàn toàn không có report full-test riêng.
* **Bằng chứng độc lập:**
  * File baseline PyTorch của pipeline cũ là [reports/pipeline_run/accuracy_reference_clean_test.json](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/reports/pipeline_run/accuracy_reference_clean_test.json). Khi đối chiếu số liệu của file này với [reports/pipeline_run/accuracy_reference_640_pt.json](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/reports/pipeline_run/accuracy_reference_640_pt.json):
    * `yolov8n`: mAP50 = `0.9098874877223397` (trùng khớp 100%)
    * `yolov9t`: mAP50 = `0.919018635031576` (trùng khớp 100%)
    * `yolov12n`: mAP50 = `0.9112156573328629` (trùng khớp 100%)
    * `yolov26n`: mAP50 = `0.916301567146553` (trùng khớp 100%)
  * Điều này chứng minh pipeline trước đây chạy hàm `execute_full_rod_pipeline.py` với tham số mặc định `imgsz=640`, sau đó lưu tên là `accuracy_reference_clean_test.json`.
  * Do đó, **chưa từng có lượt đánh giá clean test nào được thực hiện ở độ phân giải 320×320**, dẫn đến không có dữ liệu mAP50 320 thật.

### 3.5. Kiểm Tra INT8: Metadata Calibration, Backend, Số Ảnh & Metric Thật
* **INT8 Profile 320 ([int8_ptq_evaluation_report.json](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/reports/pipeline_run/int8_ptq_evaluation_report.json)):**
  * Metadata calibration: **Không có** (chỉ có nhãn chuỗi `"ultralytics_ptq_calibration"`).
  * Backend: **Không ghi nhận**.
  * Số ảnh calibration: **Thiếu hoàn toàn**.
  * Metric thật: **Thiếu hoàn toàn** (chỉ có `"sanity_check": "passed"`).
* **INT8 Profile 640 ([int8_ptq_640_accuracy_report.json](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/reports/pipeline_run/int8_ptq_640_accuracy_report.json)):**
  * Metadata calibration: Có ghi nhận phương pháp PTQ (`ultralytics_ptq_calibration`) sử dụng file dữ liệu [datasets/processed/rod_clean_v1/data.yaml](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/datasets/processed/rod_clean_v1/data.yaml).
  * Backend: Được cấu hình và thực thi qua `device="cpu"` trong Ultralytics export và `CPUExecutionProvider` trong ONNX Runtime.
  * Số ảnh calibration: Quá trình export quét qua tập validation cache (`valid/labels.cache` với 3,448 ảnh), tuy nhiên trường số lượng ảnh cụ thể không được serialize thành một key riêng biệt trong báo cáo JSON.
  * Metric thật: **Đầy đủ 100% metric thật**. Đánh giá toàn diện trên 1,615 ảnh clean test với thời gian chạy từ 135.23s đến 232.36s mỗi model:
    * `yolov8n_rod_640_onnx_int8`: mAP50 = 0.894500, mAP50–95 = 0.716012, Recall = 0.844578.
    * `yolov9t_rod_640_onnx_int8`: mAP50 = 0.911674, mAP50–95 = 0.746627, Recall = 0.868415.
    * `yolov12n_rod_640_onnx_int8`: mAP50 = 0.904712, mAP50–95 = 0.728785, Recall = 0.863238.
    * `yolov26n_rod_640_onnx_int8`: mAP50 = 0.906152, mAP50–95 = 0.746168, Recall = 0.862729.

### 3.6. Độ Suy Giảm (Deltas) .pt → ONNX → INT8 Từ Dữ Liệu Đã Lưu (Không Chạy Lại Inference)
Toàn bộ số liệu dưới đây được trích xuất nguyên trạng từ các báo cáo JSON đã lưu trong `reports/pipeline_run/`:

#### Profile 640×640 (Đầy Đủ Bằng Chứng Thực Nghiệm):
| Model | Metric | PT Baseline (`.pt`) | ONNX FP32 | INT8 PTQ | $\Delta$ (ONNX vs PT) | $\Delta$ (INT8 vs PT) | $\Delta$ (INT8 vs ONNX) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | mAP50 | 0.909887 | 0.899036 | 0.894500 | **-0.010851** | **-0.015387** | **-0.004536** |
| | mAP50–95 | 0.741934 | 0.727998 | 0.716012 | -0.013936 | -0.025922 | -0.011986 |
| | Recall | 0.867714 | 0.862096 | 0.844578 | -0.005618 | -0.023136 | -0.017518 |
| **YOLOv9t** | mAP50 | 0.919019 | 0.913443 | 0.911674 | **-0.005576** | **-0.007345** | **-0.001769** |
| | mAP50–95 | 0.771666 | 0.763073 | 0.746627 | -0.008593 | -0.025039 | -0.016446 |
| | Recall | 0.883137 | 0.877689 | 0.868415 | -0.005448 | -0.014722 | -0.009274 |
| **YOLO12n** | mAP50 | 0.911216 | 0.907053 | 0.904712 | **-0.004163** | **-0.006504** | **-0.002341** |
| | mAP50–95 | 0.760760 | 0.751665 | 0.728785 | -0.009095 | -0.031975 | -0.022880 |
| | Recall | 0.887948 | 0.874808 | 0.863238 | -0.013140 | -0.024710 | -0.011570 |
| **YOLO26n** | mAP50 | 0.916302 | 0.906423 | 0.906152 | **-0.009879** | **-0.010150** | **-0.000271** |
| | mAP50–95 | 0.771100 | 0.753135 | 0.746168 | -0.017965 | -0.024932 | -0.006967 |
| | Recall | 0.888778 | 0.867570 | 0.862729 | -0.021208 | -0.026049 | -0.004841 |

*Nhận xét:*
* Quá trình chuyển đổi `.pt` → `ONNX FP32` ở 640 làm giảm mAP50 từ **0.42% đến 1.09%**.
* Quá trình lượng tử hóa `ONNX FP32` → `INT8 PTQ` chỉ làm giảm thêm rất ít: YOLO26n giảm **0.027%**, YOLOv9t giảm **0.18%**, YOLO12n giảm **0.23%**, YOLOv8n giảm **0.45%**.
* Mức độ suy giảm tổng thể `.pt` → `INT8 PTQ` được kiểm soát nghiêm ngặt dưới ngưỡng cho phép: cao nhất là YOLOv8n (-1.54% mAP50), thấp nhất là YOLO12n (-0.65% mAP50).

#### Profile 320×320:
* Không thể tính toán delta từ dữ liệu đã lưu vì không có báo cáo đánh giá mAP clean test ở 320 cho bất kỳ runtime nào (`missing_evidence`).

### 3.7. Kiểm Tra Tính Toàn Vẹn: SHA-256, Đường Dẫn, Taxonomy, Adapters, Raspberry Pi
1. **Kiểm tra File & SHA-256:**
   * Cả **28/28 artifact files** (bao gồm file direct `.param`, `.bin`, `.onnx` ở cả 320 và 640) đều tồn tại đầy đủ trên đĩa cứng tại thư mục [weights/](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/weights/).
   * Kiểm tra mã băm SHA-256 đối chiếu với [weights/manifest.json](file:///home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/weights/manifest.json): **Khớp 100%**.
2. **Kiểm tra Resolution:**
   * Cả hai profile 320 và 640 đều được phân định rõ ràng trong manifest qua các trường `imgsz: 320` và `imgsz: 640`, input shape tương ứng `[1, 3, 320, 320]` và `[1, 3, 640, 640]`.
3. **Kiểm tra Taxonomy (25 ROD Classes):**
   * Đầy đủ 25 lớp đối tượng đúng chuẩn ROD Paper: `['Bike', 'Building', 'Car', 'Person', 'Stairs', 'Traffic sign', 'Electrical Pole', 'Road', 'Motorcycle', 'Dustbin', 'Dog', 'Manhole', 'Tree', 'Guard rail', 'Pedestrian crosswalk', 'Truck', 'Bus', 'Bench', 'Traffic Cone', 'Fire hydrant', 'Teraffic Barrel', 'Plant Pot', 'Electrical Box', 'Chair', 'Bicycle Rack']`.
   * Thứ tự index 0..24 khớp 100% giữa manifest, data.yaml và bộ pre/post-processing.
4. **Kiểm tra Output Adapters:**
   * `yolov8n`, `yolov9t`, `yolov12n`: Đầu ra tensor 3D raw `[1, 29, 8400]` (ở 640) và `[1, 29, 2100]` (ở 320) $\rightarrow$ Cấu hình adapter: **`"raw"`**.
   * `yolov26n`: Tích hợp sẵn End-to-End NMS trong ONNX graph, đầu ra tensor `[1, 300, 6]` $\rightarrow$ Cấu hình adapter: **`"e2e_nms"`**.
5. **Kiểm tra Trạng Thái Raspberry Pi:**
   * Toàn bộ 7 trường benchmark Pi trong `manifest.json` đều được giữ nguyên trạng thái đóng băng chờ phần cứng thật:
     * `latency_p50_ms`: `"Pending"`
     * `latency_p95_ms`: `"Pending"`
     * `fps`: `"Pending"`
     * `peak_ram_mb`: `"Pending"`
     * `cpu_temperature_c`: `"Pending"`
     * `throttling`: `"Pending"`
     * `edge_pareto_selection`: `"Pending — no Pi benchmark data yet"`
   * Tuyệt đối không giả lập hay bịa đặt số liệu phần cứng Pi.

---

## 4. Các Lệnh Đề Xuất Chạy Sau Khi Cần Bổ Sung Bằng Chứng (Không Tự Chạy)

Theo đúng quy tắc của đợt audit, kiểm toán viên **chỉ liệt kê các lệnh cần thiết** để lấp đầy các khoảng trống dữ liệu (`missing_evidence`) và xác minh native NCNN (`proxy`), **tuyệt đối không tự kích hoạt các lệnh này**:

### 1. Đánh giá độ chính xác thực tế cho Profile 320×320 trên tập test clean:
```bash
# Đánh giá baseline PT, ONNX FP32 và INT8 PTQ ở độ phân giải 320 trên tập rod_clean_v1/test
python scripts/run_640_profile.py --imgsz 320
```

### 2. Đánh giá độ chính xác trực tiếp từ runtime NCNN (thay thế ONNX proxy):
```bash
# Sử dụng module kiểm thử NCNN thuần (ncnn.Net C++/Python) trên rod_clean_v1/test
python -m blind_navigation.eval_ncnn --model all --imgsz 640 --data datasets/processed/rod_clean_v1/data.yaml
python -m blind_navigation.eval_ncnn --model all --imgsz 320 --data datasets/processed/rod_clean_v1/data.yaml
```

### 3. Cập nhật lại cờ `accuracy_passed` và xuất bản manifest chính thức:
```bash
# Sau khi các lệnh trên hoàn tất và sinh ra báo cáo thực nghiệm, chạy lệnh audit cập nhật manifest:
python scripts/audit_and_update_manifest.py --input-manifest weights/manifest.json --reports-dir reports/pipeline_run/
```

### 4. Benchmark trực tiếp trên bo mạch phần cứng Raspberry Pi 4:
```bash
# Thực thi trên thiết bị Raspberry Pi 4 thực tế qua SSH
bash edge_pi/run_benchmarks.sh --device pi4 --models all --repeats 100
```

---

## 5. Kết Luận Kiểm Toán

1. **Profile 640×640:**
   * ONNX FP32 và ONNX INT8 đã đạt chuẩn kiểm định cao nhất (**`verified`**), có đầy đủ báo cáo thực nghiệm 1,615 ảnh, sai số và độ suy giảm so với `.pt` được kiểm soát rất tốt.
   * NCNN FP32 640 được xác nhận là **`proxy`** từ ONNX FP32; việc gán `accuracy_passed: true` trong manifest hiện tại là **`incorrect_flag`** nếu xét theo tiêu chuẩn native NCNN validation.
2. **Profile 320×320:**
   * Toàn bộ 12 artifact ở profile 320 đều bị **`missing_evidence`** về mAP trên tập test thật, và cờ `accuracy_passed: true` trong manifest là **`incorrect_flag`** (do chỉ dựa trên cosine similarity 10 ảnh và dummy tensor sanity check).
3. **Tính toàn vẹn kỹ thuật:**
   * File model, mã băm SHA-256, adapter taxonomy 25 lớp và trạng thái `Pending` của Raspberry Pi được bảo toàn nguyên vẹn 100%, không bị sai lệch cấu trúc hay rò rỉ dữ liệu giả lập.
