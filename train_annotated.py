#!/usr/bin/env python3
"""
train_annotated.py – Build a YOLO OBB dataset from annotations and launch training.

Usage:
    python3 train_annotated.py

Reads annotations from ./annotation_output/labels/ + ./annotation_output/images/
Splits 80/20 train/val, writes a YOLO dataset structure, then calls:
    yolo obb train ...
"""
import os, shutil, random, yaml, sys, glob

LBL_DIR    = os.path.join("annotation_output", "labels")
IMG_DIR    = os.path.join("annotation_output", "images")
DS_DIR     = "putter_dataset"
MODEL_BASE = "yolov8n-obb.pt"   # nano OBB – fast to train

def build_dataset():
    pairs = []
    for lbl_path in sorted(glob.glob(os.path.join(LBL_DIR, "*.txt"))):
        base = os.path.splitext(os.path.basename(lbl_path))[0]
        # Try .jpg, .jpeg, .png
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            candidate = os.path.join(IMG_DIR, base + ext)
            if os.path.isfile(candidate):
                img_path = candidate
                break
        if img_path:
            pairs.append((img_path, lbl_path))

    if len(pairs) < 2:
        print(f"[train] Need at least 2 annotated images in {LBL_DIR}/")
        return False

    random.shuffle(pairs)
    split = max(1, int(len(pairs) * 0.8))
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:] or pairs[:1]   # at least 1 val sample

    for subset, subset_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = os.path.join(DS_DIR, "images", subset)
        lbl_dir = os.path.join(DS_DIR, "labels", subset)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for ip, lp in subset_pairs:
            shutil.copy(ip, img_dir)
            shutil.copy(lp, lbl_dir)

    data_yaml = {
        "path":  os.path.abspath(DS_DIR),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["putter_head"],
    }
    yaml_path = os.path.join(DS_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"[train] Dataset: {len(train_pairs)} train / {len(val_pairs)} val")
    print(f"[train] Config:  {yaml_path}")
    return yaml_path

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[train] ultralytics not installed.  pip install ultralytics")
        sys.exit(1)

    yaml_path = build_dataset()
    if not yaml_path:
        sys.exit(1)

    model = YOLO(MODEL_BASE)
    print(f"[train] Starting training with {MODEL_BASE} ...")
    model.train(
        data    = yaml_path,
        task    = "obb",
        epochs  = 80,
        imgsz   = 640,
        batch   = 8,
        name    = "putter_obb",
        project = "runs/obb",
        exist_ok= True,
    )
    best = os.path.join("runs", "obb", "putter_obb", "weights", "best.pt")
    print(f"\n[train] Done!  Best model: {best}")
    print("[train] Update _MODEL_PATH in putter_live.py to use it.")

if __name__ == "__main__":
    main()
