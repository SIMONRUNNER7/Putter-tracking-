"""
YOLOv8 custom putter detector training script.

This script trains a YOLOv8 model to detect golf putters in video frames.
It dramatically improves detection quality over the pure-CV fallback methods,
especially under challenging lighting, shadow, and diverse putter shapes.

=== Dataset requirements ===

Minimum recommended dataset:
  - 500 images (ideally 1000+)
  - Labeled bounding boxes around the entire putter (head + visible shaft)
  - Diverse conditions: indoor/outdoor, different green types, lighting angles
  - Both blade and mallet putter styles

Recommended tooling for labeling:
  - Roboflow (https://roboflow.com) — free tier available, exports YOLO format
  - CVAT (https://cvat.ai) — open-source alternative
  - LabelImg — local labeling tool

=== Dataset structure (YOLO format) ===

data/
  putter_dataset/
    images/
      train/   ← 80% of your images
      val/     ← 20% of your images
    labels/
      train/   ← matching .txt annotation files
      val/
    data.yaml

Each .txt file contains one line per object:
  <class_id> <x_center> <y_center> <width> <height>
  (all values normalised 0–1 relative to image size)

For putter detection we have one class:
  0: putter

=== Augmentation recommendations ===

The training script applies these augmentations automatically via Ultralytics:
  - Horizontal / vertical flip
  - Rotation ±10°
  - Brightness / contrast jitter
  - Blur (simulates out-of-focus / motion blur)
  - Mosaic mixing (helps with small objects)

Additional augmentations to add manually for golf-specific robustness:
  - Random shadows (simulates real-course lighting)
  - Colour temperature shift (indoor lighting vs. sunlight)
  - Gaussian noise (simulates low camera quality)

=== Usage ===

  pip install ultralytics
  python training/train_detector.py \\
      --data data/putter_dataset/data.yaml \\
      --epochs 100 \\
      --imgsz 640 \\
      --output runs/putter_v1

After training, the best model weights will be at:
  runs/putter_v1/weights/best.pt

Load them in the app via:
  Analysis → Use YOLO Model → select best.pt
"""

import argparse
import os
import sys
from pathlib import Path


def create_dataset_yaml(dataset_dir: str, yaml_path: str) -> None:
    """
    Auto-generate data.yaml if it doesn't exist.
    Assumes standard YOLO directory structure under dataset_dir.
    """
    content = f"""# PutterTrack Pro – Putter detection dataset config
path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val

nc: 1
names:
  0: putter
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"[train] Created dataset YAML at {yaml_path}")


def train(
    data_yaml: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    model_base: str = "yolov8n.pt",   # n=nano, s=small, m=medium
    output_dir: str = "runs/putter",
    device: str = "cpu",
) -> Path:
    """
    Fine-tune YOLOv8 on the putter dataset.
    Returns path to the best weights file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.  Run: pip install ultralytics")
        sys.exit(1)

    print(f"[train] Loading base model: {model_base}")
    model = YOLO(model_base)

    print(f"[train] Starting training for {epochs} epochs…")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=output_dir,
        name="putter_detector",
        patience=20,          # early stopping
        save=True,
        save_period=10,
        # Augmentation params tuned for golf video
        degrees=10,           # rotation
        translate=0.05,       # translation
        scale=0.3,            # zoom
        flipud=0.0,           # no vertical flip (gravity matters)
        fliplr=0.5,           # horizontal flip OK (left/right hand)
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.1,
        hsv_h=0.015,          # hue jitter (simulates lighting variation)
        hsv_s=0.5,            # saturation jitter
        hsv_v=0.4,            # value jitter (shadows / highlights)
        verbose=True,
        workers=2,            # reduce dataloader threads (default 8 causes OOM on MPS)
        cache=False,          # do not cache dataset in RAM
    )

    best_weights = Path(output_dir) / "putter_detector" / "weights" / "best.pt"
    print(f"\n[train] Done!  Best weights: {best_weights}")
    print("[train] Load in PutterTrack Pro via: Analysis → Use YOLO Model")
    return best_weights


def validate(model_path: str, data_yaml: str) -> None:
    """Run validation and print mAP metrics."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    print(f"\n[validate] mAP50:    {metrics.box.map50:.4f}")
    print(f"[validate] mAP50-95: {metrics.box.map:.4f}")
    print(f"[validate] Precision: {metrics.box.mp:.4f}")
    print(f"[validate] Recall:    {metrics.box.mr:.4f}")


def export_model(model_path: str, format: str = "onnx") -> None:
    """
    Export trained model to ONNX (for edge deployment, faster inference).
    Supports: onnx, torchscript, coreml (macOS), tflite
    """
    from ultralytics import YOLO
    model = YOLO(model_path)
    model.export(format=format)
    print(f"[export] Exported to {format}")


def prepare_augmented_shadows(image_dir: str, output_dir: str, n: int = 3) -> None:
    """
    Apply random shadow augmentations to images in image_dir and save
    to output_dir. Helps model generalise to real-course lighting.
    n = number of augmented copies per image.
    """
    import cv2
    import numpy as np
    import random

    os.makedirs(output_dir, exist_ok=True)
    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for i in range(n):
            aug = img.copy()
            # Random polygon shadow
            num_vertices = random.randint(4, 8)
            pts = np.array([
                [random.randint(0, w), random.randint(0, h)]
                for _ in range(num_vertices)
            ], dtype=np.int32)
            mask = np.ones_like(aug, dtype=np.float32)
            cv2.fillPoly(mask, [pts], (random.uniform(0.3, 0.7),) * 3)
            aug = (aug.astype(np.float32) * mask).astype(np.uint8)
            out_name = img_path.stem + f"_shadow{i}" + img_path.suffix
            cv2.imwrite(str(Path(output_dir) / out_name), aug)

    print(f"[augment] Saved {len(image_paths) * n} shadow-augmented images to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 putter detector for PutterTrack Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # --- train sub-command
    train_p = sub.add_parser("train", help="Train the model")
    train_p.add_argument("--data",    required=True, help="Path to data.yaml")
    train_p.add_argument("--epochs",  type=int, default=100)
    train_p.add_argument("--imgsz",   type=int, default=640)
    train_p.add_argument("--batch",   type=int, default=16)
    train_p.add_argument("--model",   default="yolov8n.pt",
                         help="Base YOLO model (n/s/m/l/x)")
    train_p.add_argument("--output",  default="runs/putter")
    train_p.add_argument("--device",  default="cpu",
                         help="'cpu', '0' (GPU 0), 'mps' (Apple Silicon)")

    # --- validate sub-command
    val_p = sub.add_parser("validate", help="Validate trained model")
    val_p.add_argument("--model", required=True)
    val_p.add_argument("--data",  required=True)

    # --- export sub-command
    exp_p = sub.add_parser("export", help="Export model to ONNX/CoreML/etc.")
    exp_p.add_argument("--model",  required=True)
    exp_p.add_argument("--format", default="onnx",
                       choices=["onnx", "torchscript", "coreml", "tflite"])

    # --- augment sub-command
    aug_p = sub.add_parser("augment-shadows", help="Add shadow augmentations")
    aug_p.add_argument("--input",  required=True, help="Source image directory")
    aug_p.add_argument("--output", required=True, help="Output directory")
    aug_p.add_argument("--n",      type=int, default=3, help="Copies per image")

    # --- init sub-command (create dataset scaffold)
    init_p = sub.add_parser("init-dataset", help="Create empty dataset scaffold")
    init_p.add_argument("--dir", default="data/putter_dataset")

    args = parser.parse_args()

    if args.command == "train":
        train(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            model_base=args.model,
            output_dir=args.output,
            device=args.device,
        )

    elif args.command == "validate":
        validate(args.model, args.data)

    elif args.command == "export":
        export_model(args.model, args.format)

    elif args.command == "augment-shadows":
        prepare_augmented_shadows(args.input, args.output, args.n)

    elif args.command == "init-dataset":
        d = args.dir
        for sub_dir in [
            "images/train", "images/val",
            "labels/train", "labels/val",
        ]:
            os.makedirs(os.path.join(d, sub_dir), exist_ok=True)
        create_dataset_yaml(d, os.path.join(d, "data.yaml"))
        print(f"[init] Dataset scaffold created at {d}")
        print("  → Add your labeled images to images/train and images/val")
        print("  → Add YOLO-format .txt labels to labels/train and labels/val")
        print("  → Then run: python training/train_detector.py train --data", d + "/data.yaml")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
