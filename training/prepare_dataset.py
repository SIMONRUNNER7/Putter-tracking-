"""
Prépare le dataset YOLO à partir des annotations faites avec annotate.py.

  annotation_output/images/  +  annotation_output/labels/
          ↓
  data/putter_dataset/
    images/train/   images/val/
    labels/train/   labels/val/
    data.yaml

Usage:
    python3 training/prepare_dataset.py
    python3 training/prepare_dataset.py --annotated annotation_output --out data/putter_dataset --split 0.8
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def prepare(
    annotated_dir: str = "annotation_output",
    out_dir: str = "data/putter_dataset",
    split: float = 0.8,
    seed: int = 42,
) -> None:
    ann  = Path(annotated_dir)
    imgs = ann / "images"
    lbls = ann / "labels"

    if not imgs.exists():
        print(f"[ERROR] Dossier introuvable : {imgs}")
        print("        Lance d'abord :  python3 annotate.py data/raw_images/")
        return

    # Récupère toutes les images qui ont un label non-vide (= annotées, pas skippées)
    labeled: list[tuple[Path, Path]] = []
    skipped = 0
    missing_label = 0

    for img_path in sorted(imgs.glob("*.jpg")) + sorted(imgs.glob("*.png")):
        lbl_path = lbls / (img_path.stem + ".txt")
        if not lbl_path.exists():
            missing_label += 1
            continue
        if lbl_path.stat().st_size == 0:   # skip explicite
            skipped += 1
            continue
        labeled.append((img_path, lbl_path))

    print(f"[prepare] Images annotées  : {len(labeled)}")
    print(f"[prepare] Skippées (vides) : {skipped}")
    print(f"[prepare] Sans label       : {missing_label}")

    if not labeled:
        print("[ERROR] Aucune annotation trouvée. Annote d'abord les images.")
        return

    # Mélange reproductible
    random.seed(seed)
    random.shuffle(labeled)

    n_train = max(1, int(len(labeled) * split))
    train_pairs = labeled[:n_train]
    val_pairs   = labeled[n_train:]

    print(f"[prepare] Train : {len(train_pairs)}  |  Val : {len(val_pairs)}")

    # Crée l'arborescence
    out = Path(out_dir)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    def _copy_pair(pairs, split_name):
        for img_p, lbl_p in pairs:
            shutil.copy2(img_p, out / "images" / split_name / img_p.name)
            shutil.copy2(lbl_p, out / "labels" / split_name / lbl_p.name)

    _copy_pair(train_pairs, "train")
    _copy_pair(val_pairs,   "val")

    # data.yaml
    yaml_content = f"""# PutterTrack — dataset vue de dessus (tête de putter)
path: {out.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  0: putter_head
"""
    yaml_path = out / "data.yaml"
    yaml_path.write_text(yaml_content)

    print(f"\n[prepare] Dataset prêt dans : {out.resolve()}/")
    print(f"          data.yaml          : {yaml_path}")
    print(f"\nProchaine étape :")
    print(f"  python3 training/train_detector.py train \\")
    print(f"      --data {yaml_path} \\")
    print(f"      --epochs 100 --model yolov8s.pt --device mps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated", default="annotation_output")
    parser.add_argument("--out",       default="data/putter_dataset")
    parser.add_argument("--split",     type=float, default=0.8,
                        help="Proportion train (défaut 0.8 = 80/20)")
    args = parser.parse_args()
    prepare(args.annotated, args.out, args.split)


if __name__ == "__main__":
    main()
