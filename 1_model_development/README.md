# 🧠 Module 1: Model Development & Optimization
> **Không gian huấn luyện, đánh giá và nén mô hình Trí Tuệ Nhân Tạo cho thiết bị biên**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-grey.svg)](https://onnxruntime.ai/)
[![NCNN](https://img.shields.io/badge/Tencent-NCNN-orange.svg)](https://github.com/Tencent/ncnn)

---

## 📌 1. Giới thiệu tổng quan

Module **`1_model_development`** chịu trách nhiệm toàn bộ vòng đời phát triển mô hình thị giác máy tính cục bộ (Local Edge Vision) của hệ thống điều hướng người khiếm thị:
1. **Thu thập & Tiền xử lý dữ liệu:** Chuẩn hóa dữ liệu vật cản trong nhà và đô thị (người, bàn ghế, phương tiện, cầu thang, cửa,...).
2. **Huấn luyện & Fine-tuning:** Huấn luyện các mô hình Object Detection siêu nhẹ dòng YOLO (YOLOv8n, YOLOv11n) nhằm đạt sự cân bằng giữa độ chính xác ($mAP_{50}$) và số lượng tham số.
3. **Nén & Tối ưu hóa (Quantization & Export):** Chuyển đổi mô hình PyTorch sang các định dạng suy luận tối ưu trên phần cứng biên (Raspberry Pi 4 - ARM Cortex-A72):
   - **ONNX Runtime (FP32/FP16):** Đơn giản, tính tương thích cao, đa luồng CPU linh hoạt.
   - **Tencent NCNN (FP16 Packing):** Tối ưu hóa bộ nhớ cache, khai thác tập chỉ thị SIMD ARM NEON để đạt tốc độ FPS cao nhất trên chip nhúng không có GPU rời.

---

## 📂 2. Cấu trúc thư mục

```text
1_model_development/
├── datasets/                    # Thư mục lưu trữ bộ dữ liệu huấn luyện (COCO, Roboflow, Custom)
│   ├── images/                  # Ảnh gốc (train / val / test)
│   └── labels/                  # Bounding box định dạng YOLO txt (class x_center y_center width height)
│
├── notebooks/                   # Jupyter Notebooks thực nghiệm & phân tích dữ liệu (EDA)
│   ├── 01_eda_indoor_hazards.ipynb   # Trực quan hóa phân bố vật cản và kích thước BBox
│   └── 02_benchmark_models.ipynb     # Đánh giá so sánh mAP giữa các biến thể nano/small
│
├── scripts/                     # Các kịch bản huấn luyện, validation và nén mô hình
│   ├── train.py                 # Script huấn luyện với cấu hình siêu tham số
│   ├── validate.py              # Đánh giá mAP@50-95 và Confusion Matrix trên tập kiểm thử
│   ├── export_onnx.py           # Xuất mô hình PyTorch sang ONNX (tích hợp onnxsim)
│   └── convert_to_ncnn.sh       # Script tự động chuyển đổi ONNX sang NCNN (Param + Bin)
│
└── weights/                     # Trọng số mô hình ở các định dạng khác nhau
    ├── yolov8n.pt               # Trọng số gốc PyTorch
    ├── yolov8n.onnx             # Model ONNX chuẩn hóa fixed shape 640x640
    ├── yolov8n.param            # Cấu trúc đồ thị mạng NCNN
    ├── yolov8n.bin              # Trọng số nhị phân NCNN (FP16 Packed)
    ├── yolov8s.param            # Phiên bản small NCNN
    └── yolov8s.bin
```

---

## 🎯 3. Bộ dữ liệu & Danh mục đối tượng nhận diện

Hệ thống tập trung vào **80 lớp đối tượng chuẩn COCO**, trong đó ưu tiên cao nhất các nhóm đối tượng liên quan trực tiếp đến an toàn di chuyển của người khiếm thị:

| Nhóm đối tượng | Các nhãn đại diện | Mức độ nguy cơ | Ứng dụng điều hướng |
| :--- | :--- | :---: | :--- |
| **Giao thông & Di động** | `person`, `car`, `motorcycle`, `bus`, `truck`, `bicycle` | 🔴 **Rất cao (1.0)** | Cảnh báo dừng khẩn cấp hoặc né ngay lập tức khi xuất hiện ở làn giữa |
| **Động vật** | `dog`, `cat`, `horse`, `cow` | 🟠 **Cao (0.7 - 0.85)** | Vật cản có khả năng chuyển động bất ngờ |
| **Vật cản tĩnh trong nhà** | `chair`, `couch`, `dining table`, `bed`, `refrigerator`, `toilet` | 🟡 **Trung bình (0.7 - 0.8)** | Hướng dẫn lách tránh qua trái/phải theo phân làn |
| **Vật dụng cá nhân / Nhỏ** | `backpack`, `bottle`, `cell phone`, `laptop`, `book` | 🟢 **Thấp (0.3 - 0.5)** | Hỗ trợ nhận biết đồ đạc xung quanh |

---

## 🛠 4. Quy trình Huấn luyện & Tối ưu hóa

### 4.1. Huấn luyện mô hình cơ sở (Training)
Huấn luyện mô hình YOLOv8 Nano trên tập dữ liệu vật cản:

```bash
# Huấn luyện thông qua Ultralytics CLI
yolo detect train \
  data=datasets/indoor_hazards.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/train \
  name=yolov8n_indoor
```

### 4.2. Xuất mô hình sang định dạng ONNX
Để chạy ổn định trên runtime của thiết bị biên, đồ thị ONNX cần được cố định kích thước (`dynamic=False`) và tinh giản các node thừa bằng `onnxsim`:

```bash
python scripts/export_onnx.py --weights weights/yolov8n.pt --imgsz 640
```

Nội dung script trích xuất:
```python
from ultralytics import YOLO

model = YOLO("weights/yolov8n.pt")
# simplify=True tự động kích hoạt onnxsim loại bỏ các phép toán lặp
model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
```

### 4.3. Chuyển đổi sang Tencent NCNN (FP16 Optimization)
NCNN là framework suy luận hàng đầu cho thiết bị nhúng chạy chip ARM (như Raspberry Pi 4). Quy trình chuyển đổi gồm:
1. Đưa ONNX qua công cụ `pnnx` hoặc `onnx2ncnn`.
2. Tối ưu đồ thị với `ncnnoptimize` và kích hoạt nén nửa dấu chấm động (FP16):

```bash
# Bước 1: Chuyển đổi ONNX sang NCNN
onnx2ncnn weights/yolov8n.onnx weights/yolov8n.param weights/yolov8n.bin

# Bước 2: Tối ưu đồ thị và nén FP16
ncnnoptimize weights/yolov8n.param weights/yolov8n.bin \
             weights/yolov8n_opt.param weights/yolov8n_opt.bin 65536
```

---

## 📊 5. Kết quả Benchmark trên Raspberry Pi 4 (4GB RAM)

Thực nghiệm đo đạc độ trễ suy luận trung bình (Inference Latency) và FPS trên vi xử lý ARM Cortex-A72 (Broadcom BCM2711, 4 nhân 1.5GHz):

| Mô hình | Input Size | Định dạng | Định lượng | Latency (Inference) | FPS Thực tế | RAM Usage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | $640 \times 640$ | PyTorch CPU | FP32 | ~480 ms | 2.1 FPS | ~650 MB |
| **YOLOv8n** | $640 \times 640$ | ONNX Runtime | FP32 | ~240 ms | 4.2 FPS | ~280 MB |
| **YOLOv8n** | $320 \times 320$ | ONNX Runtime | FP32 | ~95 ms | 10.5 FPS | ~190 MB |
| **YOLOv8n** | $320 \times 320$ | **NCNN (NEON)** | **FP16** | **~58 ms** | **~17.2 FPS** | **~95 MB** |
| **YOLOv8s** | $320 \times 320$ | NCNN (NEON) | FP16 | ~142 ms | 7.0 FPS | ~210 MB |

> **Nhận xét kỹ thuật:**
> - Kích thước đầu vào $320 \times 320$ ở chế độ **NCNN FP16** mang lại tốc độ phản xạ vượt trội (> 17 FPS), hoàn toàn đáp ứng ngưỡng thời gian thực cho người đi bộ.
> - Kích thước $640 \times 640$ qua **ONNX Runtime** phù hợp khi cần độ chi tiết cao và kiểm thử nhanh trên máy tính PC/Laptop có hỗ trợ tăng tốc CUDA hoặc đa luồng CPU mạnh.

---

## 🚀 6. Hướng dẫn tái hiện (Reproducibility)

### Bước 1: Cài đặt công cụ huấn luyện
```bash
uv pip install ultralytics onnx onnxsim ncnn
```

### Bước 2: Kiểm tra mô hình ONNX
```bash
python -c "import onnx; m = onnx.load('weights/yolov8n.onnx'); onnx.checker.check_model(m); print('ONNX Valid!')"
```
