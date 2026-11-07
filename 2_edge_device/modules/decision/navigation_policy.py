# 2_edge_device/modules/decision/navigation_policy.py
"""
Sinh thông điệp điều hướng bằng tiếng Việt dựa trên Top-1 target và vị trí làn.

Logic phân làn:
    frame_width chia 3 làn đều nhau:
        Trái  : cx < frame_width / 3
        Giữa  : frame_width/3 <= cx < frame_width*2/3
        Phải  : cx >= frame_width*2/3

Ưu tiên cảnh báo:
    1. Làn giữa (trực tiếp cản đường) → Hướng né hoặc dừng lại.
    2. Làn trái → Thông báo vật cản trái.
    3. Làn phải → Thông báo vật cản phải.
"""
from typing import List, Dict, Any, Tuple, Optional


# Bảng dịch nhãn COCO sang tiếng Việt tự nhiên
LABEL_VI: Dict[str, str] = {
    "person": "người",
    "bicycle": "xe đạp",
    "car": "ô tô",
    "motorcycle": "xe máy",
    "bus": "xe buýt",
    "truck": "xe tải",
    "train": "tàu",
    "dog": "con chó",
    "cat": "con mèo",
    "horse": "con ngựa",
    "cow": "con bò",
    "chair": "ghế",
    "couch": "ghế sofa",
    "dining table": "bàn",
    "bed": "giường",
    "toilet": "nhà vệ sinh",
    "tv": "màn hình",
    "refrigerator": "tủ lạnh",
    "bench": "băng ghế",
    "fire hydrant": "trụ cứu hỏa",
    "stop sign": "biển dừng",
    "backpack": "ba lô",
    "umbrella": "ô dù",
    "handbag": "túi xách",
    "suitcase": "vali",
    "bottle": "chai nước",
    "cup": "ly",
    "cell phone": "điện thoại",
    "laptop": "máy tính",
    "book": "quyển sách",
}


def _vi_name(class_name: str) -> str:
    return LABEL_VI.get(class_name, class_name)


def classify_lane(cx: float, frame_width: int) -> str:
    """
    Phân làn theo tọa độ tâm ngang của vật thể.

    Returns: 'left' | 'center' | 'right'
    """
    lane_w = frame_width / 3.0
    if cx < lane_w:
        return "left"
    elif cx < lane_w * 2.0:
        return "center"
    else:
        return "right"


def get_lanes(
    tracked: List[Dict[str, Any]],
    frame_width: int
) -> Tuple[List[str], List[str], List[str]]:
    """
    Phân loại tất cả vật thể đang track vào các làn.

    Returns:
        (left_names, center_names, right_names) – danh sách tên tiếng Việt theo từng làn.
    """
    left, center, right = [], [], []
    for obj in tracked:
        x1, _, x2, _ = obj["bbox"]
        cx = (x1 + x2) / 2.0
        lane = classify_lane(cx, frame_width)
        name = _vi_name(obj["class_name"])
        if lane == "left":
            left.append(name)
        elif lane == "center":
            center.append(name)
        else:
            right.append(name)
    return left, center, right


def generate_alert(
    top1: Optional[Dict[str, Any]],
    tracked: List[Dict[str, Any]],
    frame_width: int
) -> str:
    """
    Sinh thông điệp cảnh báo điều hướng dựa trên Top-1 và phân làn.

    Args:
        top1: Vật thể nguy hiểm nhất (dict từ PriorityScorer.top1()).
        tracked: Toàn bộ danh sách vật thể đang track trong frame.
        frame_width: Chiều rộng ảnh (pixel).

    Returns:
        Chuỗi tiếng Việt mô tả hành động cần thực hiện.
        Trả về chuỗi rỗng nếu không phát hiện vật cản đáng lo.
    """
    if not top1:
        return ""

    x1, _, x2, _ = top1["bbox"]
    cx = (x1 + x2) / 2.0
    lane = classify_lane(cx, frame_width)
    name = _vi_name(top1["class_name"])

    # Lấy thông tin tất cả làn để quyết định hướng né
    left_names, center_names, right_names = get_lanes(tracked, frame_width)

    if lane == "center":
        if not right_names or (right_names == [name] and not left_names):
            return f"Có {name} phía trước, né sang phải"
        elif not left_names or (left_names == [name] and not right_names):
            return f"Có {name} phía trước, né sang trái"
        else:
            return f"Dừng lại, có {name} chắn phía trước"

    elif lane == "left":
        return f"Cảnh báo, có {name} bên trái"

    else:  # right
        return f"Cảnh báo, có {name} bên phải"
