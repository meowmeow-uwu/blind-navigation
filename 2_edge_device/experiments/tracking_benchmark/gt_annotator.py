"""
gt_annotator.py - Công cụ Gán nhãn Ground-Truth Bán tự động cho Video Benchmark (T4: PBL6)
===========================================================================================
Tạo tệp Ground-Truth chuẩn (`frame_id, gt_id, x1, y1, x2, y2, class_name`) để tính
MOTA, IDF1 và IDSW thực sự.

Tính năng giúp gán nhãn nhanh gấp 10 lần:
- Chạy gợi ý trước bằng YOLOv8 NCNN + ByteTrack.
- Mở giao diện trực quan theo từng frame:
  + [SPACE] / [N]: Chấp nhận frame hiện tại và sang frame kế.
  + Click chuột vào hộp + nhấn phím [1..9]: Sửa đúng gt_id (tự đổi liên hoàn cho track_id đó ở các frame sau!).
  + Click chuột phải vào hộp (hoặc click + phím [D]): Xóa hộp phát hiện nhầm (False Positive).
  + Kéo thả chuột để vẽ thêm hộp bị YOLO bỏ sót (ví dụ vật bị che khuất một phần).
  + [S] hoặc [Q]: Lưu tệp CSV Ground-Truth.
- Chế độ `--empty-gt`: Tự tạo file GT rỗng cho Kịch bản 6 (hành lang không vật cản).
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

EDGE_ROOT = Path(__file__).resolve().parents[2]
if str(EDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGE_ROOT))

from modules.vision_local.yolo_ncnn_runner import YoloNcnnRunner
from modules.tracking.byte_tracker import LightweightTracker, TrackedItem


def save_gt_csv(gt_path: Path, frame_boxes: Dict[int, List[dict]]):
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame_id", "gt_id", "x1", "y1", "x2", "y2", "class_name"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fid in sorted(frame_boxes.keys()):
            for box in frame_boxes[fid]:
                writer.writerow({
                    "frame_id": fid,
                    "gt_id": box["gt_id"],
                    "x1": int(box["bbox"][0]),
                    "y1": int(box["bbox"][1]),
                    "x2": int(box["bbox"][2]),
                    "y2": int(box["bbox"][3]),
                    "class_name": box["class_name"],
                })
    print(f"💾 Đã lưu Ground-Truth ({sum(len(v) for v in frame_boxes.values())} boxes) -> {gt_path}")


def prefill_proposals(video_path: Path) -> Tuple[List, Dict[int, List[dict]]]:
    param_path = EDGE_ROOT / "modules/vision_local/models/yolov8n.param"
    bin_path = EDGE_ROOT / "modules/vision_local/models/yolov8n.bin"
    detector = YoloNcnnRunner(str(param_path), str(bin_path), target_size=320, conf_thresh=0.20)
    TrackedItem.reset_counter(0)
    tracker = LightweightTracker(max_age=12, min_hits=1, iou_threshold=0.25)

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    proposals: Dict[int, List[dict]] = {}
    fid = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
        dets = detector.detect(frame)
        tracks = tracker.update(dets)
        proposals[fid] = [
            {"gt_id": int(t.track_id), "bbox": list(t.bbox), "class_name": t.class_name}
            for t in tracks
        ]
        fid += 1
    cap.release()
    return frames, proposals


def annotate_interactive(video_path: Path, gt_path: Path):
    print(f"⏳ Đang đọc video và chạy YOLO gợi ý khung nhãn: {video_path}")
    frames, proposals = prefill_proposals(video_path)
    if not frames:
        print("❌ Video rỗng!")
        return

    total = len(frames)
    cur_fid = 0
    selected_idx = -1
    drawing = False
    draw_start = (0, 0)
    draw_curr = (0, 0)
    next_gt_id = 1

    # Tính next_gt_id lớn nhất ban đầu
    for boxes in proposals.values():
        for b in boxes:
            next_gt_id = max(next_gt_id, int(b["gt_id"]) + 1)

    def mouse_cb(event, x, y, flags, param):
        nonlocal selected_idx, drawing, draw_start, draw_curr, next_gt_id
        boxes = proposals.get(cur_fid, [])

        if event == cv2.EVENT_LBUTTONDOWN:
            # Kiểm tra xem có click trúng hộp nào không
            hit = -1
            for idx, b in enumerate(boxes):
                x1, y1, x2, y2 = b["bbox"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    hit = idx
                    break
            if hit != -1:
                selected_idx = hit
            else:
                drawing = True
                draw_start = (x, y)
                draw_curr = (x, y)
                selected_idx = -1

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            draw_curr = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x1 = min(draw_start[0], x)
            y1 = min(draw_start[1], y)
            x2 = max(draw_start[0], x)
            y2 = max(draw_start[1], y)
            if x2 - x1 > 10 and y2 - y1 > 10:
                boxes.append({"gt_id": next_gt_id, "bbox": [x1, y1, x2, y2], "class_name": "obstacle"})
                selected_idx = len(boxes) - 1
                next_gt_id += 1

        elif event == cv2.EVENT_RBUTTONDOWN:
            for idx, b in enumerate(boxes):
                x1, y1, x2, y2 = b["bbox"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    boxes.pop(idx)
                    selected_idx = -1
                    break

    win_name = "PBL6 Ground-Truth Annotator"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_cb)

    print("\nHƯỚNG DẪN PHÍM TẮT:")
    print("  [SPACE] / [N]: Frame kế | [P]: Frame trước")
    print("  Click hộp + [1..9]: Đổi gt_id (tự đồng bộ các frame sau!)")
    print("  Click hộp + [D] (hoặc Chuột phải): Xóa hộp nhiễu")
    print("  Kéo chuột trái: Vẽ hộp bị thiếu (khi vật bị che khuất)")
    print("  [S]: Lưu ngay | [Q]: Lưu & Thoát\n")

    while True:
        vis = frames[cur_fid].copy()
        boxes = proposals.setdefault(cur_fid, [])

        for idx, b in enumerate(boxes):
            x1, y1, x2, y2 = b["bbox"]
            is_sel = (idx == selected_idx)
            color = (0, 255, 255) if is_sel else (0, 220, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3 if is_sel else 2)
            label = f"GT#{b['gt_id']} {b['class_name']}"
            cv2.putText(vis, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if drawing:
            cv2.rectangle(vis, draw_start, draw_curr, (255, 160, 0), 2)

        cv2.rectangle(vis, (0, 0), (640, 28), (20, 20, 20), -1)
        cv2.putText(
            vis,
            f"Frame {cur_fid}/{total - 1} | Objects: {len(boxes)} | [SPACE] Next [P] Prev [1-9] SetID [D] Del [Q] Save&Quit",
            (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1
        )
        cv2.imshow(win_name, vis)

        key = cv2.waitKey(25) & 0xFF
        if key in [ord("q"), 27]:
            save_gt_csv(gt_path, proposals)
            break
        elif key == ord("s"):
            save_gt_csv(gt_path, proposals)
        elif key in [ord(" "), ord("n")]:
            selected_idx = -1
            cur_fid = min(total - 1, cur_fid + 1)
        elif key == ord("p"):
            selected_idx = -1
            cur_fid = max(0, cur_fid - 1)
        elif key in [ord("d"), 8, 127] and 0 <= selected_idx < len(boxes):
            boxes.pop(selected_idx)
            selected_idx = -1
        elif ord("1") <= key <= ord("9") and 0 <= selected_idx < len(boxes):
            new_id = key - ord("0")
            old_id = boxes[selected_idx]["gt_id"]
            # Tự động đổi ID liên hoàn cho các frame tiếp theo nếu đang mang old_id
            for f_future in range(cur_fid, total):
                for fb in proposals.get(f_future, []):
                    if fb["gt_id"] == old_id:
                        fb["gt_id"] = new_id
            print(f"✏️  Frame {cur_fid}: Đổi GT#{old_id} -> GT#{new_id} (từ frame {cur_fid} trở đi)")

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Công cụ gán nhãn Ground-Truth cho Benchmark Video")
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn tới file .mp4")
    parser.add_argument("--out-gt", type=str, required=True, help="Đường dẫn lưu file _gt.csv")
    parser.add_argument("--empty-gt", action="store_true", help="Tạo GT rỗng cho kịch bản hành lang trống (scen6_empty)")
    args = parser.parse_args()

    video_path = Path(args.video)
    gt_path = Path(args.out_gt)

    if args.empty_gt:
        cap = cv2.VideoCapture(str(video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_id", "gt_id", "x1", "y1", "x2", "y2", "class_name"])
            writer.writeheader()
        print(f"✅ Đã tạo Ground-Truth rỗng ({n_frames} frames không có vật cản) -> {gt_path}")
    else:
        annotate_interactive(video_path, gt_path)


if __name__ == "__main__":
    main()

