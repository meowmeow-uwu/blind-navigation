"""Tests for dataset validation pipeline."""

from pathlib import Path
from scripts.validate_dataset import validate_dataset


def test_dataset_validation_passes():
    dataset_dir = Path("datasets/processed/rod_v1")
    manifest_dir = Path("datasets/manifests")

    report = validate_dataset(dataset_dir, manifest_dir)

    assert report["status"] == "passed"
    assert report["total_images"] > 0
    assert report["total_boxes"] > 0
    assert report["invalid_annotations_count"] == 0
    assert not report["split_leakage_detected"]
