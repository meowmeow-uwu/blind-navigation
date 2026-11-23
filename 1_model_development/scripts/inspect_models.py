"""Inspect checkpoint taxonomy and basic properties of validated models.

Usage (from 1_model_development/):
    python scripts/inspect_models.py
    python scripts/inspect_models.py --models yolov8n yolo11n --output benchmarks/summaries/taxonomy_check.json

Models are taken from weights/manifest.json (never by scanning folders).
Checks: class_count == 80 and identical class ids for person/bicycle/car/motorcycle.
If taxonomies differ, models cannot be compared on pretrained recall without a class mapping.
Exit code is 1 if any check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    COCO_KEY_CLASSES,
    EXPECTED_CLASS_COUNT,
    STATUS_VALIDATED,
    SUMMARIES_DIR,
    class_names_hash,
    load_manifest,
    resolve_manifest_checkpoint,
    write_json,
)


def inspect_checkpoint(name: str, ckpt: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    names = {int(k): str(v) for k, v in model.names.items()}
    stride = getattr(model.model, "stride", None)
    stride_max = int(stride.max()) if stride is not None else None
    default_imgsz = (getattr(model, "overrides", {}) or {}).get("imgsz")
    return {
        "model": name,
        "task": model.task,
        "parameters": int(sum(p.numel() for p in model.model.parameters())),
        "class_count": len(names),
        "first_10_classes": [names[i] for i in sorted(names)[:10]],
        "stride": stride_max,
        "default_imgsz": default_imgsz,
        "checkpoint_size_mb": round(ckpt.stat().st_size / (1024 * 1024), 2),
        "class_names_hash": class_names_hash(names),
        "key_class_ids": {c: next((i for i, n in names.items() if n == c), None) for c in COCO_KEY_CLASSES},
    }


def print_table(rows: list[dict]) -> None:
    headers = ["model", "task", "params", "classes", "stride", "imgsz", "size_MB", "first_10_classes"]
    table = [
        [
            r["model"],
            r["task"],
            f"{r['parameters']:,}",
            str(r["class_count"]),
            str(r["stride"]),
            str(r["default_imgsz"]),
            f"{r['checkpoint_size_mb']:.2f}",
            ", ".join(r["first_10_classes"]),
        ]
        for r in rows
    ]
    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in table:
        print(fmt.format(*row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=None, help="Default: all validated in manifest")
    parser.add_argument("--output", type=Path, default=SUMMARIES_DIR / "taxonomy_check.json")
    args = parser.parse_args()

    manifest = load_manifest()
    if manifest is None:
        print("ERROR: weights/manifest.json not found. Run scripts/download_models.py first.", file=sys.stderr)
        return 2

    requested = args.models or list(manifest["models"])
    rows: list[dict] = []
    skipped: dict[str, str] = {}
    for name in requested:
        entry = manifest["models"].get(name)
        if entry is None:
            skipped[name] = "not in manifest"
            continue
        if entry["status"] != STATUS_VALIDATED:
            skipped[name] = entry["status"]
            continue
        ckpt = resolve_manifest_checkpoint(manifest, name)
        if ckpt is None or not ckpt.is_file():
            skipped[name] = f"checkpoint missing on disk: {ckpt}"
            continue
        try:
            rows.append(inspect_checkpoint(name, ckpt))
        except Exception as exc:  # noqa: BLE001
            skipped[name] = f"inspect failed: {type(exc).__name__}: {exc}"

    if not rows:
        print("ERROR: no validated model to inspect.", file=sys.stderr)
        for name, why in skipped.items():
            print(f"  {name}: {why}", file=sys.stderr)
        return 2

    print_table(rows)
    for name, why in skipped.items():
        print(f"\nSKIPPED {name}: {why}")

    # --- taxonomy checks -------------------------------------------------------
    problems: list[str] = []
    for r in rows:
        if r["class_count"] != EXPECTED_CLASS_COUNT:
            problems.append(f"{r['model']}: class_count={r['class_count']}, expected {EXPECTED_CLASS_COUNT}")
        for cls, expected_id in COCO_KEY_CLASSES.items():
            got = r["key_class_ids"][cls]
            if got != expected_id:
                problems.append(f"{r['model']}: '{cls}' has class_id={got}, expected {expected_id}")

    reference = rows[0]
    identical = all(r["class_names_hash"] == reference["class_names_hash"] for r in rows)
    ids_agree = all(r["key_class_ids"] == reference["key_class_ids"] for r in rows)

    print("\nTaxonomy checks:")
    print(f"  class_count == {EXPECTED_CLASS_COUNT} for all models: "
          f"{all(r['class_count'] == EXPECTED_CLASS_COUNT for r in rows)}")
    for cls, expected_id in COCO_KEY_CLASSES.items():
        ids = {r["model"]: r["key_class_ids"][cls] for r in rows}
        print(f"  {cls:10s} same class_id across models: {len(set(ids.values())) == 1} {ids}")
    print(f"  identical class_names_hash across models: {identical}")

    if identical and not problems:
        verdict = "OK: taxonomies are identical; pretrained recall can be compared directly."
    else:
        verdict = (
            "MISMATCH: taxonomies differ from COCO-80 / between models. These models "
            "CANNOT be compared on pretrained recall directly without a class mapping."
        )
        for p in problems:
            print(f"  PROBLEM: {p}")
        if not ids_agree:
            print("  PROBLEM: key class ids differ between models")
    print(f"\n{verdict}")

    write_json(
        args.output if args.output.is_absolute() else Path.cwd() / args.output,
        {
            "models": rows,
            "skipped": skipped,
            "identical_taxonomy": identical,
            "problems": problems,
            "verdict": verdict,
        },
    )
    print(f"Report written: {args.output}")
    return 0 if identical and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
