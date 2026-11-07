# 2_edge_device/tests/test_decision.py
"""
Bộ kiểm thử cho toàn bộ module Decision (Giai đoạn 3):
  - PriorityScorer
  - TrackStateManager (cooldown, re-alert, eviction)
  - NavigationPolicy (lane classify, alert message)
"""
import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EDGE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, EDGE_DIR)

from modules.decision.priority_scorer import PriorityScorer
from modules.decision.track_state_manager import TrackStateManager
from modules.decision.navigation_policy import (
    classify_lane, get_lanes, generate_alert
)

# Đường dẫn file cấu hình
WEIGHTS_PATH = os.path.join(EDGE_DIR, "configs/class_weights.yaml")

FRAME_W, FRAME_H = 640, 480


def _make_obj(
    class_name: str,
    x1: int, y1: int, x2: int, y2: int,
    track_id: int = 1,
    conf: float = 0.9
):
    return {
        "track_id": track_id,
        "class_name": class_name,
        "class_id": 0,
        "bbox": [x1, y1, x2, y2],
        "confidence": conf,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. PriorityScorer
# ──────────────────────────────────────────────────────────────────────────────
def test_priority_scorer():
    print("[TEST] PriorityScorer...")
    scorer = PriorityScorer(WEIGHTS_PATH, w_class=0.4, w_position=0.35, w_size=0.25)

    # Người (c=1.0) ở trung tâm, bbox lớn → điểm cao nhất
    person_center = _make_obj("person", 220, 100, 420, 400)  # cx=320 = center
    s1 = scorer.score(person_center, FRAME_W, FRAME_H)

    # Chai nước (c=0.35) ở rìa phải → điểm thấp
    bottle_edge = _make_obj("bottle", 580, 200, 620, 280)
    s2 = scorer.score(bottle_edge, FRAME_W, FRAME_H)

    print(f"  person_center score: {s1:.4f}")
    print(f"  bottle_edge   score: {s2:.4f}")
    assert s1 > s2, "Person ở giữa phải có điểm cao hơn bottle ở rìa"

    # Top-1 chọn đúng
    top = scorer.top1([person_center, bottle_edge], FRAME_W, FRAME_H)
    assert top is not None and top["class_name"] == "person"

    # Rank giảm dần
    ranked = scorer.rank([bottle_edge, person_center], FRAME_W, FRAME_H)
    assert ranked[0]["class_name"] == "person"

    print("  -> test_priority_scorer: PASS!")


# ──────────────────────────────────────────────────────────────────────────────
# 2. TrackStateManager
# ──────────────────────────────────────────────────────────────────────────────
def test_track_state_manager():
    print("[TEST] TrackStateManager...")
    mgr = TrackStateManager(cooldown_seconds=1.0, re_alert_delta=0.25, max_ttl=5.0)

    # Track mới → luôn phát
    assert mgr.should_alert(1, 0.6), "Track mới phải được cảnh báo"

    # Lần 2 ngay lập tức với cùng score → không phát (cooldown)
    assert not mgr.should_alert(1, 0.6), "Phải bị chặn bởi cooldown"

    # Priority tăng đáng kể (Preemption) → phát lại dù chưa hết cooldown
    assert mgr.should_alert(1, 0.6 + 0.30), "Priority tăng 0.30 >= delta 0.25 → phải alert"

    # Hết cooldown → phát lại
    time.sleep(1.05)
    assert mgr.should_alert(1, 0.5), "Đã hết cooldown → phải alert"

    # Track độc lập (ID khác nhau → cooldown riêng biệt)
    assert mgr.should_alert(99, 0.7), "Track ID 99 chưa từng xuất hiện → phải alert"
    assert not mgr.should_alert(99, 0.7), "Track ID 99 trong cooldown"

    # Evict TTL
    mgr2 = TrackStateManager(cooldown_seconds=10.0, re_alert_delta=0.5, max_ttl=0.1)
    mgr2.should_alert(100, 0.5)
    assert mgr2.active_track_count == 1
    time.sleep(0.15)
    removed = mgr2.evict_stale()
    assert removed == 1 and mgr2.active_track_count == 0, "Eviction TTL thất bại"

    print("  -> test_track_state_manager: PASS!")


# ──────────────────────────────────────────────────────────────────────────────
# 3. NavigationPolicy
# ──────────────────────────────────────────────────────────────────────────────
def test_navigation_policy():
    print("[TEST] NavigationPolicy...")

    # Phân làn
    assert classify_lane(100, 640) == "left"   # 100 < 640/3 ≈ 213
    assert classify_lane(320, 640) == "center"
    assert classify_lane(550, 640) == "right"

    # Vật thể ở giữa, làn trái và phải trống → né phải
    person_center = _make_obj("person", 220, 100, 420, 400, track_id=1)
    person_center["priority_score"] = 0.9
    alert = generate_alert(person_center, [person_center], FRAME_W)
    assert "phải" in alert or "trái" in alert or "Dừng" in alert, f"Alert sai: {alert}"
    print(f"  Trường hợp giữa: '{alert}'")

    # Vật thể ở trái
    obj_left = _make_obj("chair", 50, 100, 150, 300, track_id=2)
    obj_left["priority_score"] = 0.6
    alert2 = generate_alert(obj_left, [obj_left], FRAME_W)
    assert "trái" in alert2, f"Phải báo bên trái: {alert2}"
    print(f"  Trường hợp trái:  '{alert2}'")

    # Vật thể ở phải
    obj_right = _make_obj("bicycle", 520, 100, 620, 300, track_id=3)
    obj_right["priority_score"] = 0.7
    alert3 = generate_alert(obj_right, [obj_right], FRAME_W)
    assert "phải" in alert3, f"Phải báo bên phải: {alert3}"
    print(f"  Trường hợp phải:  '{alert3}'")

    # Không có vật thể
    alert_empty = generate_alert(None, [], FRAME_W)
    assert alert_empty == "", "Không có vật thể → alert trống"

    print("  -> test_navigation_policy: PASS!")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_priority_scorer()
    test_track_state_manager()
    test_navigation_policy()
    print("\n✅ Tất cả test Giai đoạn 3 PASS!")
