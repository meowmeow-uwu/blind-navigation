"""
evaluate_results.py - Bộ tính toán Metrics chuẩn MOTChallenge & Báo cáo Thực nghiệm (T4: PBL6)
=============================================================================================
Đọc log CSV từ `run_benchmark.py` và đối chiếu với Ground-Truth CSV để tính toán hoàn toàn
tự động, minh bạch các thước đo học thuật thực tế:
- Thành phần Re-ID sử dụng
- MOTA (%) [Bernardin & Stiefelhagen 2008]
- IDF1 (%) [Ristani et al., ECCVW 2016]
- ID Switches (IDSW)
- FPS thực tế & Latency (ms)
- Tỷ lệ nói lặp (lần/phút) trên cùng một vật thể Ground-Truth
- Tỷ lệ cảnh báo nhầm trên hành lang trống (Kịch bản 6)

Xuất bảng tổng hợp ra Terminal và file Markdown `results/tracking_benchmark_report.md`.
"""

import os
import sys
import csv
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

EDGE_ROOT = Path(__file__).resolve().parents[2]
if str(EDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGE_ROOT))

from modules.tracking.byte_tracker import compute_iou


def _max_weight_match(weight_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Tìm ghép cặp cực đại trên ma trận trọng số dương."""
    if weight_matrix.size == 0:
        return []
    if HAS_SCIPY:
        r_ind, c_ind = linear_sum_assignment(-weight_matrix)
        return [(int(r), int(c)) for r, c in zip(r_ind, c_ind) if weight_matrix[r, c] > 0]
    work = weight_matrix.copy()
    pairs = []
    while True:
        r, c = np.unravel_index(np.argmax(work), work.shape)
        if work[r, c] <= 0:
            break
        pairs.append((int(r), int(c)))
        work[r, :] = -1
        work[:, c] = -1
    return pairs


def load_ground_truth(gt_csv_path: Path) -> Dict[int, List[dict]]:
    gt_by_frame: Dict[int, List[dict]] = defaultdict(list)
    if not gt_csv_path.exists():
        return gt_by_frame
    with open(gt_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["frame_id"])
            gt_by_frame[fid].append({
                "gt_id": int(row["gt_id"]),
                "bbox": (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
                "class_name": row["class_name"],
            })
    return gt_by_frame


def load_tracker_log(log_csv_path: Path):
    """Đọc file log CSV và nhóm theo từng frame_id."""
    frames: Dict[int, dict] = {}
    reid_info = "Không"
    tracker_name = ""

    with open(log_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["frame_id"])
            tracker_name = row["tracker_name"]
            reid_info = row.get("reid_info", "Không")

            entry = frames.setdefault(fid, {
                "timestamp": float(row["timestamp"]),
                "det_ms": float(row["det_time_ms"]),
                "track_ms": float(row["track_time_ms"]),
                "proc_ms": float(row["processing_time_ms"]),
                "hyps": [],
                "raw_audio_events": [],
            })

            tid = int(row["track_id"])
            audio_ev = row.get("audio_event", "none")
            if audio_ev in ("guidance", "emergency"):
                entry["raw_audio_events"].append(audio_ev)

            if tid >= 0 and row["bbox"].strip():
                parts = [int(float(v)) for v in row["bbox"].split(",")]
                entry["hyps"].append({
                    "track_id": tid,
                    "bbox": tuple(parts),
                    "class_name": row["class_name"],
                    "distance": float(row["distance_estimate"]) if row["distance_estimate"] else None,
                    "audio_event": audio_ev,
                })

    return tracker_name, reid_info, frames


def evaluate_scenario(log_csv_path: Path, gt_csv_path: Path, iou_thresh: float = 0.50) -> dict:
    """Tính đầy đủ bộ chỉ số MOT + HCI cho 1 kịch bản x 1 tracker."""
    tracker_name, reid_info, log_frames = load_tracker_log(log_csv_path)
    gt_by_frame = load_ground_truth(gt_csv_path)

    total_frames = len(log_frames)
    all_fids = sorted(log_frames.keys())
    if not all_fids:
        return {}

    timestamps = [log_frames[fid]["timestamp"] for fid in all_fids]
    dt_step = (timestamps[-1] - timestamps[0]) / max(1, len(timestamps) - 1) if len(timestamps) > 1 else (1.0 / 15.0)
    duration_s = max(1e-3, (timestamps[-1] - timestamps[0]) + dt_step)
    duration_min = duration_s / 60.0

    tp = 0
    fp = 0
    fn = 0
    idsw = 0
    total_gt = 0
    total_hyp = 0

    last_tid_for_gt: Dict[int, int] = {}
    gt_counts: Dict[int, int] = defaultdict(int)
    hyp_counts: Dict[int, int] = defaultdict(int)
    overlap_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    # Thống kê HCI âm thanh
    guidance_per_gt: Dict[int, int] = defaultdict(int)
    total_guidance_utterances = 0
    false_alerts = 0

    for fid in all_fids:
        frame_data = log_frames[fid]
        hyps = frame_data["hyps"]
        gts = gt_by_frame.get(fid, [])

        total_gt += len(gts)
        total_hyp += len(hyps)

        for g in gts:
            gt_counts[g["gt_id"]] += 1
        for h in hyps:
            hyp_counts[h["track_id"]] += 1

        if len(gts) == 0:
            false_alerts += len(frame_data["raw_audio_events"])

        for h in hyps:
            if h["audio_event"] == "guidance":
                total_guidance_utterances += 1

        # Khớp GT và Hypothesis theo IoU >= iou_thresh
        matched_gt = set()
        matched_hyp = set()
        if gts and hyps:
            iou_mat = np.zeros((len(gts), len(hyps)), dtype=np.float64)
            for i, g in enumerate(gts):
                for j, h in enumerate(hyps):
                    val = compute_iou(g["bbox"], h["bbox"])
                    iou_mat[i, j] = val if val >= iou_thresh else 0.0

            pairs = _max_weight_match(iou_mat)
            for gi, hi in pairs:
                g_id = gts[gi]["gt_id"]
                t_id = hyps[hi]["track_id"]
                matched_gt.add(gi)
                matched_hyp.add(hi)
                tp += 1
                overlap_counts[(g_id, t_id)] += 1

                if g_id in last_tid_for_gt and last_tid_for_gt[g_id] != t_id:
                    idsw += 1
                last_tid_for_gt[g_id] = t_id

                if hyps[hi]["audio_event"] == "guidance":
                    guidance_per_gt[g_id] += 1

        fn += len(gts) - len(matched_gt)
        fp += len(hyps) - len(matched_hyp)

        # Kiểm tra bất kỳ âm thanh guidance phát trên FP box (không khớp GT)
        for j, h in enumerate(hyps):
            if j not in matched_hyp and h["audio_event"] == "guidance":
                if len(gts) > 0:
                    # Nếu khung cảnh có GT nhưng bbox này không khớp GT -> vẫn tính phát thừa
                    guidance_per_gt[-999] += 1

    # Tính IDF1 toàn cục
    unique_gts = sorted(gt_counts.keys())
    unique_hyps = sorted(hyp_counts.keys())
    idtp = 0
    if unique_gts and unique_hyps:
        w_mat = np.zeros((len(unique_gts), len(unique_hyps)), dtype=np.float64)
        for r, gid in enumerate(unique_gts):
            for c, tid in enumerate(unique_hyps):
                w_mat[r, c] = overlap_counts.get((gid, tid), 0)
        global_pairs = _max_weight_match(w_mat)
        for r, c in global_pairs:
            idtp += int(w_mat[r, c])

    idfn = total_gt - idtp
    idfp = total_hyp - idtp
    idf1 = (2.0 * idtp / (2.0 * idtp + idfp + idfn) * 100.0) if (2 * idtp + idfp + idfn) > 0 else None
    mota = ((1.0 - (fn + fp + idsw) / total_gt) * 100.0) if total_gt > 0 else None

    # Số lần nói lặp = tổng số lần nói trên cùng 1 gt_id trừ đi lần đầu tiên (min 0)
    repeated_guidance = sum(max(0, cnt - 1) for gid, cnt in guidance_per_gt.items() if gid != -999)
    repeat_rate_per_min = repeated_guidance / duration_min
    false_alert_rate_per_min = false_alerts / duration_min

    avg_det_ms = float(np.mean([log_frames[fid]["det_ms"] for fid in all_fids]))
    avg_track_ms = float(np.mean([log_frames[fid]["track_ms"] for fid in all_fids]))
    avg_proc_ms = float(np.mean([log_frames[fid]["proc_ms"] for fid in all_fids]))
    fps = 1000.0 / avg_proc_ms if avg_proc_ms > 0 else 0.0

    return {
        "tracker_name": tracker_name,
        "reid_info": reid_info,
        "frames": total_frames,
        "duration_s": duration_s,
        "total_gt": total_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "idsw": idsw,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "mota": mota,
        "idf1": idf1,
        "avg_det_ms": avg_det_ms,
        "avg_track_ms": avg_track_ms,
        "avg_proc_ms": avg_proc_ms,
        "fps": fps,
        "repeated_guidance": repeated_guidance,
        "total_guidance": total_guidance_utterances,
        "repeat_rate_per_min": repeat_rate_per_min,
        "false_alerts": false_alerts,
        "false_alert_rate_per_min": false_alert_rate_per_min,
    }


def aggregate_by_tracker(scenario_metrics: List[dict]) -> dict:
    """Tổng hợp toàn cục (Micro-average chuẩn MOTChallenge) qua tất cả các kịch bản."""
    total_gt = sum(m["total_gt"] for m in scenario_metrics)
    total_tp = sum(m["tp"] for m in scenario_metrics)
    total_fp = sum(m["fp"] for m in scenario_metrics)
    total_fn = sum(m["fn"] for m in scenario_metrics)
    total_idsw = sum(m["idsw"] for m in scenario_metrics)
    total_idtp = sum(m["idtp"] for m in scenario_metrics)
    total_idfp = sum(m["idfp"] for m in scenario_metrics)
    total_idfn = sum(m["idfn"] for m in scenario_metrics)

    total_frames = sum(m["frames"] for m in scenario_metrics)
    total_dur_min = sum(m["duration_s"] for m in scenario_metrics) / 60.0
    total_rep = sum(m["repeated_guidance"] for m in scenario_metrics)
    total_false = sum(m["false_alerts"] for m in scenario_metrics)

    weighted_proc = sum(m["avg_proc_ms"] * m["frames"] for m in scenario_metrics) / max(1, total_frames)
    weighted_track = sum(m["avg_track_ms"] * m["frames"] for m in scenario_metrics) / max(1, total_frames)
    fps = 1000.0 / weighted_proc if weighted_proc > 0 else 0.0

    mota = ((1.0 - (total_fn + total_fp + total_idsw) / total_gt) * 100.0) if total_gt > 0 else None
    idf1 = (2.0 * total_idtp / (2.0 * total_idtp + total_idfp + total_idfn) * 100.0) if (2 * total_idtp + total_idfp + total_idfn) > 0 else None

    return {
        "tracker_name": scenario_metrics[0]["tracker_name"],
        "reid_info": scenario_metrics[0]["reid_info"],
        "mota": mota,
        "idf1": idf1,
        "idsw": total_idsw,
        "fps": fps,
        "avg_proc_ms": weighted_proc,
        "avg_track_ms": weighted_track,
        "repeat_rate_per_min": total_rep / max(1e-6, total_dur_min),
        "false_alert_rate_per_min": total_false / max(1e-6, total_dur_min),
        "total_frames": total_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Đánh giá & Tổng hợp kết quả thực nghiệm Object Tracking")
    parser.add_argument("--logs-dir", type=str, default=str(Path(__file__).resolve().parent / "results/logs"))
    parser.add_argument("--gt-dir", type=str, default=str(Path(__file__).resolve().parent / "data/ground_truth"))
    parser.add_argument("--out-md", type=str, default=str(Path(__file__).resolve().parent / "results/tracking_benchmark_report.md"))
    parser.add_argument("--iou-thresh", type=float, default=0.50)
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    gt_dir = Path(args.gt_dir)
    out_md = Path(args.out_md)

    log_files = sorted(logs_dir.glob("*__*.csv"))
    if not log_files:
        print(f"❌ Không tìm thấy tệp log *__*.csv nào tại {logs_dir}")
        return

    grouped_by_tracker: Dict[str, List[dict]] = defaultdict(list)
    per_scenario_rows = []

    for lf in log_files:
        stem = lf.stem
        scen_id, trk_name = stem.split("__", 1)
        gt_file = gt_dir / f"{scen_id}_gt.csv"
        if not gt_file.exists():
            print(f"⚠️  Thiếu file Ground-Truth cho '{scen_id}' tại {gt_file} -> Bỏ qua kịch bản này.")
            continue

        metrics = evaluate_scenario(lf, gt_file, iou_thresh=args.iou_thresh)
        if not metrics:
            continue
        metrics["scenario_id"] = scen_id
        grouped_by_tracker[trk_name].append(metrics)
        per_scenario_rows.append(metrics)

    if not grouped_by_tracker:
        print("❌ Không có dữ liệu nào đủ điều kiện đánh giá (kiểm tra lại file _gt.csv).")
        return

    order = ["raw_yolo", "sort", "deepsort", "edge_bytetrack"]
    display_names = {
        "raw_yolo": "Raw YOLO (Không tracker)",
        "sort": "SORT (Bewley et al., 2016)",
        "deepsort": "DeepSORT (Wojke et al., 2017)",
        "edge_bytetrack": "Edge ByteTrack (PBL6)",
    }

    print("\n" + "=" * 115)
    print("BẢNG KẾT QUẢ THỰC NGHIỆM THỰC TẾ: ĐỐI SÁNH 4 CẤU HÌNH OBJECT TRACKING (DỮ LIỆU ĐO ĐẠC TỪ LOG)")
    print("=" * 115)
    header = f"{'Thuật toán Tracker':<28} | {'Thành phần Re-ID':<28} | {'MOTA':<7} | {'IDF1':<7} | {'IDSW':<5} | {'FPS':<8} | {'Track ms':<8} | {'Nói lặp (lần/p)':<15}"
    print(header)
    print("-" * 115)

    md_lines = [
        "# KẾT QUẢ THỰC NGHIỆM THỰC TẾ OBJECT TRACKING (TRÍCH XUẤT TỪ LOG ĐO ĐẠC)\n",
        "## 1. Bảng tổng hợp toàn cục qua tất cả kịch bản\n",
        "| Thuật toán Tracker | Thành phần Re-ID | MOTA (%) | IDF1 (%) | ID Switches (IDSW) | FPS thực tế | Latency (ms/frame) | Tỷ lệ nói lặp (lần/phút) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for key in order:
        if key not in grouped_by_tracker:
            continue
        agg = aggregate_by_tracker(grouped_by_tracker[key])
        d_name = display_names.get(key, key)
        mota_str = f"{agg['mota']:.1f}%" if agg["mota"] is not None else "—"
        idf1_str = f"{agg['idf1']:.1f}%" if agg["idf1"] is not None else "—"
        idsw_str = str(agg["idsw"])
        fps_str = f"{agg['fps']:.1f}"
        lat_str = f"{agg['avg_proc_ms']:.1f} ms"
        rep_str = f"{agg['repeat_rate_per_min']:.1f}"

        print(
            f"{d_name:<28} | {agg['reid_info']:<28} | {mota_str:<7} | {idf1_str:<7} | "
            f"{idsw_str:<5} | {fps_str:<8} | {agg['avg_track_ms']:<8.2f} | {rep_str:<15}"
        )
        md_lines.append(
            f"| **{d_name}** | {agg['reid_info']} | {mota_str} | {idf1_str} | {idsw_str} | {fps_str} FPS | {lat_str} | {rep_str} lần/phút |"
        )

    print("=" * 115)

    # Thêm chi tiết theo từng kịch bản vào Markdown
    md_lines.extend([
        "\n## 2. Chi tiết kết quả theo từng kịch bản kiểm thử\n",
        "| Kịch bản | Thuật toán | Frames | TP / FP / FN | IDSW | MOTA (%) | IDF1 (%) | FPS | Nói lặp (lần/p) | Cảnh báo nhầm (lần/p) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in sorted(per_scenario_rows, key=lambda x: (x["scenario_id"], order.index(x["tracker_name"]) if x["tracker_name"] in order else 99)):
        mota_s = f"{r['mota']:.1f}%" if r["mota"] is not None else "—"
        idf1_s = f"{r['idf1']:.1f}%" if r["idf1"] is not None else "—"
        md_lines.append(
            f"| `{r['scenario_id']}` | `{r['tracker_name']}` | {r['frames']} | "
            f"{r['tp']}/{r['fp']}/{r['fn']} | {r['idsw']} | {mota_s} | {idf1_s} | "
            f"{r['fps']:.1f} | {r['repeat_rate_per_min']:.1f} | {r['false_alert_rate_per_min']:.1f} |"
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"📄 Đã xuất báo cáo Markdown từ dữ liệu thực nghiệm -> {out_md}")


if __name__ == "__main__":
    main()

