# 2_edge_device/modules/decision/track_state_manager.py
"""
Quản lý trạng thái cảnh báo riêng biệt cho từng Track ID.

Quy tắc phát cảnh báo (theo thứ tự ưu tiên):
    1. Vật thể mới: track_id lần đầu xuất hiện → Luôn cảnh báo.
    2. Priority tăng đáng kể: current_score - last_score >= delta_threshold → Báo lại ngay (Preempt).
    3. Hết cooldown riêng của track này: time.time() - last_time >= cooldown_seconds → Báo lại.
    4. Không thỏa mãn → Bỏ qua.

Hỗ trợ tự dọn dẹp (auto-evict) các track không còn xuất hiện sau max_ttl giây
để tránh rò rỉ bộ nhớ khi chạy lâu dài.
"""
from typing import Dict, Optional
import time
import dataclasses


@dataclasses.dataclass
class TrackRecord:
    """Bản ghi trạng thái của một Track ID."""
    last_alert_time: float    # Unix timestamp lần cảnh báo cuối
    last_priority: float      # Điểm ưu tiên của lần cảnh báo cuối
    last_seen_time: float     # Unix timestamp lần cuối cùng track này được cập nhật
    alert_count: int = 0      # Tổng số lần đã phát cảnh báo


class TrackStateManager:
    """
    Quản lý cooldown và điều kiện re-alert theo từng track_id.
    Thread-safe khi chỉ có 1 luồng gọi (pipeline đơn luồng).
    """

    def __init__(
        self,
        cooldown_seconds: float = 3.0,
        re_alert_delta: float = 0.25,
        max_ttl: float = 30.0
    ):
        """
        Args:
            cooldown_seconds: Thời gian tối thiểu (giây) giữa 2 lần cảnh báo cho cùng track_id.
            re_alert_delta: Mức tăng priority_score tối thiểu để kích hoạt cảnh báo sớm (Preempt).
            max_ttl: Thời gian (giây) sau đó nếu track không xuất hiện sẽ bị xóa khỏi bộ nhớ.
        """
        self.cooldown_seconds = cooldown_seconds
        self.re_alert_delta = re_alert_delta
        self.max_ttl = max_ttl

        self._records: Dict[int, TrackRecord] = {}

    def should_alert(self, track_id: int, priority_score: float) -> bool:
        """
        Kiểm tra xem track_id này có nên được phát cảnh báo hay không.
        Cập nhật trạng thái nội bộ nếu quyết định phát.

        Args:
            track_id: Track ID từ ByteTrack.
            priority_score: Điểm ưu tiên hiện tại từ PriorityScorer.

        Returns:
            True nếu cần phát cảnh báo, False nếu không.
        """
        now = time.monotonic()

        # ── Vật thể hoàn toàn mới ─────────────────────────────────────────────
        if track_id not in self._records:
            self._records[track_id] = TrackRecord(
                last_alert_time=now,
                last_priority=priority_score,
                last_seen_time=now,
                alert_count=1
            )
            return True

        record = self._records[track_id]
        record.last_seen_time = now

        # ── Re-alert: Priority tăng đáng kể (Preemption) ─────────────────────
        delta = priority_score - record.last_priority
        if delta >= self.re_alert_delta:
            record.last_alert_time = now
            record.last_priority = priority_score
            record.alert_count += 1
            return True

        # ── Cooldown đã hết ───────────────────────────────────────────────────
        elapsed = now - record.last_alert_time
        if elapsed >= self.cooldown_seconds:
            record.last_alert_time = now
            record.last_priority = priority_score
            record.alert_count += 1
            return True

        return False

    def touch(self, track_id: int, priority_score: float) -> None:
        """
        Chỉ cập nhật thời gian 'seen' và điểm ưu tiên của track (không phát cảnh báo).
        Dùng cho các vật thể đang track nhưng không phải Top-1.
        """
        now = time.monotonic()
        if track_id not in self._records:
            self._records[track_id] = TrackRecord(
                last_alert_time=0.0,
                last_priority=priority_score,
                last_seen_time=now,
                alert_count=0
            )
        else:
            self._records[track_id].last_seen_time = now

    def evict_stale(self) -> int:
        """
        Xóa các track đã vắng mặt quá max_ttl giây.
        Gọi định kỳ (ví dụ: mỗi 100 frame) để tránh rò rỉ bộ nhớ.

        Returns:
            Số lượng track đã bị xóa.
        """
        now = time.monotonic()
        stale_ids = [
            tid for tid, rec in self._records.items()
            if now - rec.last_seen_time > self.max_ttl
        ]
        for tid in stale_ids:
            del self._records[tid]
        return len(stale_ids)

    def get_record(self, track_id: int) -> Optional[TrackRecord]:
        return self._records.get(track_id)

    def reset(self) -> None:
        """Xóa toàn bộ trạng thái – dùng khi chuyển nguồn video."""
        self._records.clear()

    @property
    def active_track_count(self) -> int:
        return len(self._records)
