#!/usr/bin/env python3
"""Run selected PBL6 320x320 ROD models on an image, video, or camera."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_ROOT = REPO_ROOT / "pi_deployment_package"
ROD_CLASSES = [
    "Bike", "Building", "Car", "Person", "Stairs", "Traffic sign",
    "Electrical Pole", "Road", "Motorcycle", "Dustbin", "Dog", "Manhole",
    "Tree", "Guard rail", "Pedestrian crosswalk", "Truck", "Bus", "Bench",
    "Traffic Cone", "Fire hydrant", "Teraffic Barrel", "Plant Pot",
    "Electrical Box", "Chair", "Bicycle Rack",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entry(manifest: dict, model: str, backend: str, precision: str) -> dict:
    if backend == "onnx" and precision == "fp32":
        key = "onnx_fp32_320"
    elif backend == "onnx" and precision == "int8":
        key = "int8_ptq_320"
    else:
        key = "ncnn_fp32_320"
    try:
        return manifest["models"][model][key]
    except KeyError as error:
        raise ValueError(f"Missing manifest entry: {model}/{key}") from error


def load_detector(args: argparse.Namespace):
    package_root = args.package_root.resolve()
    manifest_path = package_root / "weights" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = artifact_entry(manifest, args.model, args.backend, args.precision)
    if entry.get("accuracy_passed") is not True:
        raise ValueError(f"Artifact is not marked accuracy_passed: {args.model}")

    sys.path.insert(0, str(package_root))
    if args.backend == "onnx":
        weight_dir = "onnx" if args.precision == "fp32" else "int8"
        artifact = package_root / "weights" / weight_dir / f"{args.model}_320_{args.precision}.onnx"
        expected = entry.get("sha256")
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"ONNX artifact missing or SHA-256 mismatch: {artifact}")

        import onnxruntime as ort
        detector_module = importlib.import_module("models.onnx_detector")
        ONNXDetector = detector_module.ONNXDetector
        postprocess_raw_output = detector_module.postprocess_raw_output
        preprocess = detector_module.preprocess

        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        detector = ONNXDetector(
            str(artifact), class_names=ROD_CLASSES, session_options=options
        )

        def infer(frame: np.ndarray) -> list[dict]:
            tensor, ratio, pad = preprocess(
                frame, (detector.input_height, detector.input_width)
            )
            output = detector.session.run(
                [detector.output_name],
                {detector.input_name: tensor},
            )[0]
            if detector.output_format == "end2end":
                detections = detector._postprocess_end2end(
                    output, ratio, pad, frame.shape[:2], args.conf
                )
            else:
                detections = postprocess_raw_output(
                    output, ratio, pad, frame.shape[:2], args.conf, args.iou,
                    ROD_CLASSES,
                )
            return [
                {
                    "class_id": item.class_id,
                    "class_name": item.class_name,
                    "confidence": item.confidence,
                    "bbox_xyxy": list(item.bbox_xyxy),
                }
                for item in detections
            ]

        return infer, artifact, entry.get("output_adapter", "unknown")

    if args.precision != "fp32":
        raise ValueError("NCNN supports fp32 only")

    import ncnn
    detector_module = importlib.import_module("models.onnx_detector")
    letterbox = detector_module.letterbox
    postprocess_raw_output = detector_module.postprocess_raw_output

    model_dir = package_root / "weights" / "ncnn" / f"{args.model}_ncnn_fp32"
    param_path = model_dir / "model.ncnn.param"
    bin_path = model_dir / "model.ncnn.bin"
    expected_hashes = entry.get("sha256", {})
    if not param_path.is_file() or not bin_path.is_file():
        raise FileNotFoundError(f"NCNN model files are missing: {model_dir}")
    if sha256_file(param_path) != expected_hashes.get("param"):
        raise ValueError(f"NCNN param SHA-256 mismatch: {param_path}")
    if sha256_file(bin_path) != expected_hashes.get("bin"):
        raise ValueError(f"NCNN bin SHA-256 mismatch: {bin_path}")

    net = ncnn.Net()
    net.opt.num_threads = args.threads
    net.opt.use_vulkan_compute = False
    if net.load_param(str(param_path)) != 0 or net.load_model(str(bin_path)) != 0:
        raise RuntimeError(f"Could not load NCNN model: {model_dir}")

    def infer(frame: np.ndarray) -> list[dict]:
        boxed, ratio, pad = letterbox(frame, (320, 320))
        tensor = ncnn.Mat.from_pixels(
            boxed, ncnn.Mat.PixelType.PIXEL_BGR2RGB, boxed.shape[1], boxed.shape[0]
        )
        tensor.substract_mean_normalize(
            [0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0]
        )
        extractor = net.create_extractor()
        if extractor.input("in0", tensor) != 0:
            raise RuntimeError("NCNN input binding failed")
        status, output = extractor.extract("out0")
        if status != 0:
            raise RuntimeError(f"NCNN extraction failed: {status}")
        detections = postprocess_raw_output(
            np.asarray(output), ratio, pad, frame.shape[:2], args.conf, args.iou,
            ROD_CLASSES,
        )
        return [
            {
                "class_id": item.class_id,
                "class_name": item.class_name,
                "confidence": item.confidence,
                "bbox_xyxy": list(item.bbox_xyxy),
            }
            for item in detections
        ]

    return infer, model_dir, "raw + CPU NMS"


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    for item in detections:
        x1, y1, x2, y2 = (int(value) for value in item["bbox_xyxy"])
        label = f"{item['class_name']} {item['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 80), 2)
        cv2.putText(
            frame, label, (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 80), 1,
        )
    return frame


def run(args: argparse.Namespace) -> None:
    infer, artifact, adapter = load_detector(args)
    print(json.dumps({
        "model": args.model,
        "backend": args.backend,
        "precision": args.precision,
        "input_size": 320,
        "confidence_threshold": args.conf,
        "nms_iou_threshold": args.iou,
        "output_adapter": adapter,
        "artifact": str(artifact),
    }, indent=2))

    source = int(args.source) if args.source.isdigit() else args.source
    image = cv2.imread(str(source)) if isinstance(source, str) else None
    if image is not None:
        print(json.dumps(infer(image), indent=2))
        return

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"Cannot open image/video/camera source: {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    writer = None
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = infer(frame)
            frame_index += 1
            if args.save_video:
                if writer is None:
                    output = Path(args.save_video)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    codec = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(output), codec, fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                writer.write(draw_detections(frame, detections))
            if args.display:
                cv2.imshow("PBL6 ROD inference", draw_detections(frame, detections))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if detections:
                print(json.dumps({"frame": frame_index, "detections": detections}))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
    print(f"Processed {frame_index} frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image/video path or camera index, e.g. 0")
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--model", choices=("yolov8n", "yolov9t", "yolov12n", "yolov26n"), default="yolov26n")
    parser.add_argument("--backend", choices=("onnx", "ncnn"), default="onnx")
    parser.add_argument("--precision", choices=("fp32", "int8"), default="int8")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save-video")
    args = parser.parse_args()
    if args.backend == "ncnn" and args.precision != "fp32":
        parser.error("NCNN artifacts are FP32 only")
    run(args)


if __name__ == "__main__":
    main()
