# 2_edge_device/experiments/debug_tensor.py
import ncnn
import numpy as np

net = ncnn.Net()
net.load_param("../modules/vision_local/models/yolov8n.param")
net.load_model("../modules/vision_local/models/yolov8n.bin")

# Tạo ảnh giả lập chuẩn 640x640
fake_in = ncnn.Mat(640, 640, 3)
ex = net.create_extractor()
ex.input("images", fake_in)
ret, mat_out = ex.extract("output")

arr = np.array(mat_out)
print(f"[*] NCNN Mat Dimensions: w={mat_out.w}, h={mat_out.h}, c={mat_out.c}")
print(f"[*] NumPy Array Shape: {arr.shape}")
print(f"[*] 5 giá trị đầu tiên của hàng 0:\n", arr[0, :5] if arr.ndim > 1 else arr[:5])