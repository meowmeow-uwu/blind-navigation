# 2_edge_device/experiments/export_onnx.py
import os
from ultralytics import YOLO

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(current_dir, "../modules/vision_local/models"))
    pt_path = os.path.join(model_dir, "yolov8n.pt")
    
    print(f"[INFO] Bắt đầu xuất ONNX từ: {pt_path}")
    model = YOLO(pt_path)
    output_path = model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    print(f"[SUCCESS] Đã tạo file ONNX thành công tại: {output_path}")

if __name__ == "__main__":
    main()
