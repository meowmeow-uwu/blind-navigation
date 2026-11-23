import hashlib
import json
import cv2
import numpy as np
from pathlib import Path

base_img = cv2.imread("assets/smoke_test.jpg")
h, w = base_img.shape[:2]

splits = {
    "train": [
        ("IMG_00001", 3, (0.2, 0.4, 0.15, 0.5)),      # Person
        ("IMG_00002", 16, (0.5, 0.45, 0.6, 0.6)),    # Bus
        ("IMG_00003", 2, (0.8, 0.6, 0.25, 0.3)),     # Car
        ("IMG_00004", 17, (0.35, 0.7, 0.2, 0.15)),   # Bench
    ],
    "valid": [
        ("IMG_19185", 3, (0.25, 0.42, 0.14, 0.48)),  # Person
        ("IMG_19186", 0, (0.7, 0.65, 0.2, 0.3)),     # Bike
    ],
    "test": [
        ("IMG_19187", 3, (0.18, 0.38, 0.12, 0.45)),  # Person
        ("IMG_19188", 16, (0.52, 0.48, 0.58, 0.58)), # Bus
        ("IMG_19189", 2, (0.75, 0.55, 0.22, 0.28)),  # Car
        ("IMG_19190", 19, (0.12, 0.75, 0.1, 0.2)),   # Fire hydrant
        ("IMG_19191", 4, (0.4, 0.8, 0.3, 0.25)),     # Stairs (domain-only)
    ]
}

root = Path("datasets/processed/rod_v1")

for split_name, items in splits.items():
    img_dir = root / split_name / "images"
    lbl_dir = root / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (stem, class_id, (cx, cy, bw, bh)) in enumerate(items):
        # Slightly vary each image to ensure unique content & hash
        img_copy = base_img.copy()
        cv2.putText(img_copy, f"{split_name}_{stem}", (30, 50 + idx * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out_img_path = img_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_img_path), img_copy)
        
        out_lbl_path = lbl_dir / f"{stem}.txt"
        with open(out_lbl_path, "w") as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

print("Generated rod_v1 processed sample splits successfully.")
