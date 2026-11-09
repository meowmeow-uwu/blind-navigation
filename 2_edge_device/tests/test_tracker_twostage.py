"""
test_tracker_twostage.py - Kiểm thử Giải thuật Khớp nối 2 Tầng ByteTrack (ECCV 2022)
====================================================================================
Kiểm chứng:
1. Khớp nối Tầng 1 với D_high (score >= 0.45)
2. Khớp nối Tầng 2 cứu vớt track bị che khuất với D_low (0.20 <= score < 0.45) -> GIỮ NGUYÊN ID
3. D_low chưa ghép bị loại bỏ (không tạo track rác/nhiễu nền)
4. Triệt tiêu nói lặp qua cờ 'announced'
"""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from modules.tracking.byte_tracker import LightweightTracker, TrackedItem


class TestByteTrackTwoStage(unittest.TestCase):

    def setUp(self):
        self.tracker = LightweightTracker(
            max_age=10,
            min_hits=1,
            iou_threshold=0.25,
            high_thresh=0.45,
            low_thresh=0.20
        )

    def test_stage2_low_confidence_recovery(self):
        """Khung hình 1 điểm cao tạo Track. Khung hình 2 vật bị che khuất điểm tụt còn 0.32 -> ByteTrack cứu vớt, giữ nguyên ID!"""
        # Frame 1: Detection điểm cao (0.85)
        f1_dets = [
            {'bbox': [100, 100, 200, 300], 'class_name': 'person', 'confidence': 0.85}
        ]
        tracks_f1 = self.tracker.update(f1_dets)
        self.assertEqual(len(tracks_f1), 1)
        initial_id = tracks_f1[0].track_id

        # Frame 2: Vật cản bị che khuất một phần, confidence tụt còn 0.32 (thuộc D_low)
        f2_dets = [
            {'bbox': [102, 105, 204, 305], 'class_name': 'person', 'confidence': 0.32}
        ]
        tracks_f2 = self.tracker.update(f2_dets)

        # ByteTrack Tầng 2 phải cứu vớt track này và GIỮ NGUYÊN ID
        self.assertEqual(len(tracks_f2), 1)
        self.assertEqual(tracks_f2[0].track_id, initial_id, "Track ID phải được bảo toàn khi cứu ở Tầng 2!")
        self.assertEqual(tracks_f2[0].hits, 2)

    def test_unmatched_low_confidence_dropped(self):
        """Detection điểm thấp (D_low) xuất hiện lần đầu (không khớp track cũ nào) phải bị BỎ QUA, không tạo track rác."""
        # Frame 1: Xuất hiện 1 detection điểm thấp (0.30) ở góc ảnh
        f1_dets = [
            {'bbox': [400, 300, 500, 450], 'class_name': 'chair', 'confidence': 0.30}
        ]
        tracks = self.tracker.update(f1_dets)

        # Không được tạo track mới từ D_low!
        self.assertEqual(len(tracks), 0, "D_low chưa ghép tuyệt đối không được tạo track mới!")
        self.assertEqual(len(self.tracker.tracks), 0)

    def test_anti_spam_silent_mode(self):
        """Kiểm tra cờ announced và chế độ Silent Tracking."""
        f1_dets = [
            {'bbox': [100, 100, 200, 300], 'class_name': 'chair', 'confidence': 0.90}
        ]
        tracks = self.tracker.update(f1_dets)
        trk = tracks[0]
        self.assertFalse(trk.announced)

        # Xướng âm thông báo lần đầu
        trk.mark_announced(distance_m=1.8)
        self.assertTrue(trk.announced)

        # Khung hình tiếp theo: track vẫn giữ announced = True (không nói lại)
        f2_dets = [
            {'bbox': [102, 100, 202, 300], 'class_name': 'chair', 'confidence': 0.88}
        ]
        tracks_f2 = self.tracker.update(f2_dets)
        self.assertTrue(tracks_f2[0].announced, "Trạng thái announced phải duy trì qua các frame tiếp theo!")


if __name__ == "__main__":
    unittest.main()

