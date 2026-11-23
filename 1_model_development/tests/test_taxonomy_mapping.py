"""Tests for taxonomy and class mapping."""

from pathlib import Path
from scripts.validate_class_mapping import validate_taxonomy_and_mapping


def test_taxonomy_mapping_passes():
    rod_cfg = Path("configs/datasets/rod.yaml")
    map_cfg = Path("configs/datasets/rod_to_coco.yaml")
    danger_cfg = Path("configs/danger_classes.yaml")
    dist_csv = Path("datasets/manifests/class_distribution.csv")
    out_dir = Path("benchmarks/summaries")

    report = validate_taxonomy_and_mapping(rod_cfg, map_cfg, danger_cfg, dist_csv, out_dir)

    assert report["status"] == "passed"
    assert report["total_classes"] == 25
    assert report["shared_coco_classes_count"] == 11
    assert report["domain_only_classes_count"] == 14
    assert len(report["errors"]) == 0
