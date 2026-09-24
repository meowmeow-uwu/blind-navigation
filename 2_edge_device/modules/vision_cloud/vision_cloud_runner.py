# 2_edge_device/modules/vision_cloud/vision_cloud_runner.py
import cv2
import base64
import time
from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url="http://qwen.nguyendanh.dpdns.org/v1"
)

SYSTEM_PROMPT = (
    "Bạn là một người bạn đồng hành đang dẫn đường cho một người khiếm thị. "
    "Quan sát ảnh và mô tả thật tự nhiên trong 1 đến 2 câu ngắn gọn bằng tiếng Việt:\n"
    "- Hãy gọi tên chính xác các đồ vật hoặc vật cản nổi bật nhất trước mắt (bàn, ghế, cửa, cột, người, bức tường...).\n"
    "- Chỉ rõ vị trí và lối đi an toàn nhất (ở bên trái, bên phải hay đi thẳng).\n"
    "- Tuyệt đối không đánh số thứ tự như 1, 2. Trả lời trực tiếp bằng một lời chỉ dẫn thông thường."
)

def process_and_encode_image(frame, max_size=384):
    """
    Nén ảnh về 384px để giảm kích thước token ảnh truyền qua mạng,
    đảm bảo server sinh phản hồi dưới 2 giây.
    """
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buffer).decode('utf-8')

def call_qwen_vl_server(frame):
    start_time = time.time()
    base64_image = process_and_encode_image(frame)
    
    try:
        response = client.chat.completions.create(
            model="llava:7b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.45,      # Tăng độ linh hoạt từ ngữ, tránh sinh câu rập khuôn lặp lại
            max_tokens=90,         # Đủ cho 2 câu chỉ dẫn tự nhiên
            timeout=15.0
        )
        latency = time.time() - start_time
        content = response.choices[0].message.content.strip()
        print(f"[Vision Cloud] Phản hồi trong: {latency:.2f}s | Nội dung: {content}")
        return content
    except Exception as e:
        return f"Lỗi gọi VLM Cloud: {str(e)}"