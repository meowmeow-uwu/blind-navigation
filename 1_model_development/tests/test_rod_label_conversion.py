"""Unit tests for ROD-Dataset YOLO label schema audit and polygon-to-bbox conversion."""

import csv
import json
import pytest
from pathlib import Path
from scripts.convert_rod_labels import (
    classify_line,
    convert_polygon_to_bbox,
    process_label_lines,
    run_pipeline,
    sha256_text,
)


class TestDetectionClassification:
    """Tests for detection line classification and preservation."""

    def test_valid_detection_line(self):
        line = "14 0.56953125 0.4796875 0.01953125 0.0390625"
        res = classify_line(line)
        assert res["type"] == "detection"
        assert res["token_count"] == 5
        assert res["class_id"] == 14
        assert len(res["coords"]) == 4
        assert res["error"] is None

    def test_detection_boundary_limits(self):
        # Valid boundary box
        line = "0 0.5 0.5 1.0 1.0"
        res = classify_line(line)
        assert res["type"] == "detection"
        assert res["coords"] == [0.5, 0.5, 1.0, 1.0]

    def test_detection_preserves_content_exact(self):
        line = "3 0.123456789 0.987654321 0.456789 0.654321"
        clean_lines, audit_recs, summary = process_label_lines([line], file_identifier="test.txt")
        assert len(clean_lines) == 1
        assert clean_lines[0] == line
        assert summary["category"] == "detection_only"
        assert summary["detection_lines"] == 1
        assert summary["segmentation_lines"] == 0
        assert summary["invalid_lines"] == 0


class TestPolygonSegmentation:
    """Tests for segmentation polygon classification and conversion to bounding box."""

    def test_valid_segmentation_classification(self):
        # 5 points = 1 + 2*5 = 11 tokens (from IMG_20781.txt)
        line = "14 0.3875 0.6111111109375 0.53125 0.6111111109375 0.52421875 0.5930555562500001 0.4015625 0.590277778125 0.3875 0.6111111109375"
        res = classify_line(line)
        assert res["type"] == "segmentation"
        assert res["token_count"] == 11
        assert res["class_id"] == 14
        assert len(res["coords"]) == 10
        assert res["error"] is None

    def test_polygon_minimum_vertices_n3(self):
        # 3 points = 1 + 2*3 = 7 tokens (triangle)
        triangle_line = "2 0.1 0.1 0.3 0.1 0.2 0.4"
        res = classify_line(triangle_line)
        assert res["type"] == "segmentation"
        assert res["token_count"] == 7

        cx, cy, bw, bh, fmt_line = convert_polygon_to_bbox(res["class_id"], res["coords"], precision=6)
        assert cx == pytest.approx(0.2, abs=1e-5)
        assert cy == pytest.approx(0.25, abs=1e-5)
        assert bw == pytest.approx(0.2, abs=1e-5)
        assert bh == pytest.approx(0.3, abs=1e-5)
        assert fmt_line == "2 0.200000 0.250000 0.200000 0.300000"

    def test_img_20781_polygon_conversion(self):
        # Exact conversion of IMG_20781.txt polygon line
        poly_line = "14 0.3875 0.6111111109375 0.53125 0.6111111109375 0.52421875 0.5930555562500001 0.4015625 0.590277778125 0.3875 0.6111111109375"
        res = classify_line(poly_line)
        cx, cy, bw, bh, fmt_line = convert_polygon_to_bbox(res["class_id"], res["coords"], precision=6)

        # Expected:
        # xmin = 0.3875, xmax = 0.53125 -> cx = 0.459375, bw = 0.143750
        # ymin = 0.590277778125, ymax = 0.6111111109375 -> cy = 0.600694, bh = 0.020833
        assert cx == pytest.approx(0.459375, abs=1e-5)
        assert cy == pytest.approx(0.600694, abs=1e-5)
        assert bw == pytest.approx(0.143750, abs=1e-5)
        assert bh == pytest.approx(0.020833, abs=1e-5)
        assert fmt_line == "14 0.459375 0.600694 0.143750 0.020833"

    def test_polygon_clamping_to_unit_square(self):
        # Slightly noisy coordinates outside [0, 1] clamped safely
        coords = [-0.01, 0.1, 1.05, 0.1, 0.5, 0.9]
        cx, cy, bw, bh, fmt_line = convert_polygon_to_bbox(0, coords, precision=4)
        assert cx == 0.5
        assert bw == 1.0
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= bw <= 1.0


class TestMixedFileHandling:
    """Tests for files containing both detection and segmentation lines."""

    def test_mixed_file_img_20781(self):
        raw_lines = [
            "14 0.56953125 0.4796875 0.01953125 0.0390625",
            "14 0.3875 0.6111111109375 0.53125 0.6111111109375 0.52421875 0.5930555562500001 0.4015625 0.590277778125 0.3875 0.6111111109375",
        ]
        clean_lines, audit_recs, summary = process_label_lines(raw_lines, file_identifier="test/labels/IMG_20781.txt")

        assert summary["category"] == "mixed"
        assert summary["detection_lines"] == 1
        assert summary["segmentation_lines"] == 1
        assert summary["invalid_lines"] == 0
        assert len(clean_lines) == 2

        # Line 1 unchanged detection
        assert clean_lines[0] == "14 0.56953125 0.4796875 0.01953125 0.0390625"
        # Line 2 converted bbox
        assert clean_lines[1] == "14 0.459375 0.600694 0.143750 0.020833"

        # Verify audit records
        assert len(audit_recs) == 2
        assert audit_recs[0]["type"] == "detection"
        assert audit_recs[0]["token_count"] == 5
        assert audit_recs[1]["type"] == "segmentation"
        assert audit_recs[1]["token_count"] == 11


class TestInvalidRowHandling:
    """Tests for invalid row identification, error logging, and exclusion."""

    @pytest.mark.parametrize("invalid_line,expected_error", [
        ("0", "invalid_token_count_1"),
        ("0 0.5 0.5", "invalid_token_count_3"),
        ("0 0.5 0.5 0.2", "invalid_token_count_4"),
        ("0 0.5 0.5 0.2 0.2 0.2", "invalid_token_count_6"),
        ("cat 0.5 0.5 0.2 0.2", "non_integer_class_id_cat"),
        ("-1 0.5 0.5 0.2 0.2", "negative_class_id_-1"),
        ("0 0.5 abc 0.2 0.2", "non_float_token_at_idx_2_abc"),
        ("0 0.5 0.5 0.0 0.2", "zero_or_negative_box_dimensions"),
        ("0 0.5 0.5 -0.1 0.2", "zero_or_negative_box_dimensions"),
        ("0 2.5 0.5 0.2 0.2", "coordinates_excessively_outside_unit_range"),
        ("1 0.2 0.2 0.2 0.2 0.2 0.2", "degenerate_polygon_flat_area"),  # flat polygon with 0 area
    ])
    def test_invalid_line_classification(self, invalid_line, expected_error):
        res = classify_line(invalid_line)
        assert res["type"] == "invalid"
        assert res["error"] == expected_error

    def test_invalid_row_is_logged_and_not_in_clean_lines(self):
        lines = [
            "0 0.5 0.5 0.2 0.2",       # valid detection
            "99 bad line token count",  # invalid
            "1 0.1 0.1 0.3 0.1 0.2 0.4", # valid polygon
        ]
        clean_lines, audit_recs, summary = process_label_lines(lines, file_identifier="sample.txt")

        assert summary["category"] == "invalid"
        assert summary["detection_lines"] == 1
        assert summary["segmentation_lines"] == 1
        assert summary["invalid_lines"] == 1
        assert len(summary["invalid_rows"]) == 1
        assert summary["invalid_rows"][0]["line"] == 2
        assert "bad" in summary["invalid_rows"][0]["content"]

        # Only 2 clean lines output (valid detection + converted polygon), invalid omitted
        assert len(clean_lines) == 2
        assert clean_lines[0] == "0 0.5 0.5 0.2 0.2"
        assert clean_lines[1] == "1 0.200000 0.250000 0.200000 0.300000"


class TestConversionPipeline:
    """Tests for the complete conversion and audit pipeline end-to-end."""

    def test_pipeline_on_synthetic_dataset(self, tmp_path):
        raw_root = tmp_path / "raw" / "ROD-Dataset"
        processed_root = tmp_path / "processed" / "rod_detection_v1"
        audit_csv = tmp_path / "manifests" / "rod_label_schema_audit.csv"
        patch_manifest = tmp_path / "manifests" / "patch_manifest.json"

        # Create mock raw dataset with train/valid/test
        for split in ["train", "valid", "test"]:
            (raw_root / split / "images").mkdir(parents=True, exist_ok=True)
            (raw_root / split / "labels").mkdir(parents=True, exist_ok=True)

        # Create dummy image and label files:
        # 1. Detection-only in train
        (raw_root / "train" / "images" / "IMG_001.jpg").write_bytes(b"dummy_img_1")
        (raw_root / "train" / "labels" / "IMG_001.txt").write_text("3 0.2 0.4 0.15 0.5\n")

        # 2. Segmentation-only in valid
        (raw_root / "valid" / "images" / "IMG_002.jpg").write_bytes(b"dummy_img_2")
        (raw_root / "valid" / "labels" / "IMG_002.txt").write_text("11 0.1 0.1 0.3 0.1 0.2 0.3\n")

        # 3. Mixed in test (like IMG_20781)
        (raw_root / "test" / "images" / "IMG_20781.jpg").write_bytes(b"dummy_img_3")
        (raw_root / "test" / "labels" / "IMG_20781.txt").write_text(
            "14 0.56953125 0.4796875 0.01953125 0.0390625\n"
            "14 0.3875 0.6111111109375 0.53125 0.6111111109375 0.52421875 0.5930555562500001 0.4015625 0.590277778125 0.3875 0.6111111109375\n"
        )

        # Run pipeline
        manifest = run_pipeline(
            raw_root=raw_root,
            output_root=processed_root,
            audit_csv_path=audit_csv,
            patch_manifest_path=patch_manifest,
            precision=6,
        )

        # Assert raw files remain unchanged
        assert (raw_root / "test" / "labels" / "IMG_20781.txt").read_text().startswith("14 0.56953125")

        # Assert processed files exist and are all 5-token bboxes
        clean_20781 = (processed_root / "test" / "labels" / "IMG_20781.txt").read_text().strip().split("\n")
        assert len(clean_20781) == 2
        assert len(clean_20781[0].split()) == 5
        assert len(clean_20781[1].split()) == 5

        # Assert counts in manifest
        assert manifest["file_counts"]["detection_only"] == 1
        assert manifest["file_counts"]["segmentation_only"] == 1
        assert manifest["file_counts"]["mixed"] == 1
        assert manifest["file_counts"]["total"] == 3
        assert manifest["instance_counts"]["polygons_converted"] == 2  # IMG_002 and line 2 of IMG_20781
        assert manifest["instance_counts"]["detection_lines_preserved"] == 2
        assert manifest["instance_counts"]["total_final_instances"] == 4

        # Assert audit CSV
        assert audit_csv.exists()
        with open(audit_csv, "r") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 4
            types = {r["type"] for r in reader}
            assert types == {"detection", "segmentation"}

        # Assert patch manifest JSON
        assert patch_manifest.exists()
        with open(patch_manifest, "r") as f:
            pdata = json.load(f)
            assert pdata["img_20781_txt_change"]["output_sha256"] != ""
            assert len(pdata["img_20781_txt_change"]["processed_lines"]) == 2
