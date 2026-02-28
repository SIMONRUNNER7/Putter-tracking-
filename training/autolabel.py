"""
Auto-labeler for putter product images → YOLO format.

How it works:
  1. Green background removal  — most product photos have a uniform green
     (or white/grey) background. We segment the putter using HSV masking
     or GrabCut, then fit a tight bounding box.

  2. Multi-strategy detection:
       A) HSV green mask     — handles green-grass product shots
       B) HSV white mask     — handles white studio background
       C) GrabCut refinement — refines any rough mask
       D) Saliency detection — fallback for complex backgrounds

  3. Quality filters         — rejects images where detection is uncertain
     (bbox too small/large, low mask fill, multiple disconnected blobs).

  4. YOLO label output       — writes .txt with normalised bbox coords.

Usage:
    python training/autolabel.py \
        --images data/raw_images \
        --out    data/putter_dataset \
        --review           # saves debug visualisations to data/review/

Then split into train/val:
    python training/autolabel.py split --dataset data/putter_dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Segmentation strategies
# ---------------------------------------------------------------------------

def _mask_green_bg(bgr: np.ndarray) -> np.ndarray:
    """
    Create a foreground mask by removing green/grass backgrounds.
    Returns uint8 mask (255=foreground).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Green range (grass, putting green, product background)
    lo_green  = np.array([30,  40,  40])
    hi_green  = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lo_green, hi_green)

    # Bright-green artificial turf (more saturated)
    lo_bright = np.array([35, 80, 80])
    hi_bright = np.array([85, 255, 255])
    bright_mask = cv2.inRange(hsv, lo_bright, hi_bright)

    bg_mask = cv2.bitwise_or(green_mask, bright_mask)
    fg_mask = cv2.bitwise_not(bg_mask)

    return fg_mask


def _mask_white_bg(bgr: np.ndarray) -> np.ndarray:
    """Foreground mask for white/light-grey studio backgrounds."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # White = low saturation + high value
    lo = np.array([0,   0, 180])
    hi = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lo, hi)
    return cv2.bitwise_not(white_mask)


def _mask_neutral_bg(bgr: np.ndarray) -> np.ndarray:
    """Foreground mask for grey/dark/mixed studio backgrounds (low saturation)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Neutral/grey = low saturation, any brightness
    lo = np.array([0,   0,  60])
    hi = np.array([180, 50, 230])
    neutral_mask = cv2.inRange(hsv, lo, hi)
    return cv2.bitwise_not(neutral_mask)


def _mask_grabcut(bgr: np.ndarray, rough_mask: np.ndarray) -> np.ndarray:
    """
    Refine a rough foreground mask with GrabCut.
    rough_mask: uint8, 255=likely foreground, 0=likely background.
    """
    h, w = bgr.shape[:2]
    # Find bounding box of rough mask
    coords = cv2.findNonZero(rough_mask)
    if coords is None or len(coords) < 50:
        return rough_mask

    x, y, bw, bh = cv2.boundingRect(coords)
    # Add padding
    pad = 10
    x  = max(0, x - pad)
    y  = max(0, y - pad)
    bw = min(w - x, bw + 2*pad)
    bh = min(h - y, bh + 2*pad)

    if bw < 20 or bh < 20:
        return rough_mask

    gc_mask  = np.zeros((h, w), dtype=np.uint8)
    gc_mask[:] = cv2.GC_BGD   # probable background

    # Mark rough foreground region
    fg_coords = np.argwhere(rough_mask > 127)
    for (fy, fx) in fg_coords[::4]:     # subsample for speed
        gc_mask[fy, fx] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(bgr, gc_mask, (x, y, bw, bh),
                    bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return rough_mask

    refined = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)
    return refined


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in the mask."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels <= 1:
        return mask
    # Skip label 0 (background)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological cleanup of a binary mask."""
    k5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k15)
    mask = cv2.dilate(mask, k5, iterations=1)
    return mask


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------

def _grabcut_center_rect(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    GrabCut using a central rectangle as foreground hint.
    Works on any background type. Returns uint8 mask or None.
    """
    h, w = bgr.shape[:2]
    mx = max(10, int(w * 0.15))
    my = max(10, int(h * 0.15))
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    gc_mask  = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(bgr, gc_mask, rect, bgd_model, fgd_model, 4,
                    cv2.GC_INIT_WITH_RECT)
        result = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        ratio = cv2.countNonZero(result) / (h * w)
        if 0.03 <= ratio <= 0.88:
            return result
    except cv2.error:
        pass
    return None


def _mask_to_bbox(
    mask: np.ndarray, h: int, w: int, skip_quality: bool = False,
) -> Optional[tuple[float, float, float, float]]:
    """Extract YOLO bbox from a binary mask, or None if quality checks fail."""
    mask = _largest_component(mask)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    x, y, bw, bh = cv2.boundingRect(coords)

    if not skip_quality:
        if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
            return None
        if bw / w > 0.99 or bh / h > 0.99:
            return None
        # Add small padding
        pad_x = int(bw * 0.04)
        pad_y = int(bh * 0.04)
        x  = max(0, x - pad_x)
        y  = max(0, y - pad_y)
        bw = min(w - x, bw + 2 * pad_x)
        bh = min(h - y, bh + 2 * pad_y)

    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    nw = bw / w
    nh = bh / h
    return (cx, cy, nw, nh)


def detect_putter_bbox(
    bgr: np.ndarray,
    use_grabcut: bool = True,
) -> tuple[Optional[tuple[float, float, float, float]], str]:
    """
    Detect putter bounding box in an image.
    Returns ((x_center, y_center, width, height), strategy_name).
    Each strategy is tried end-to-end; if the bbox fails quality
    checks, we fall through to the next strategy.

    Strategy order:
      1. HSV background removal (green / white / neutral)
      2. GrabCut with central rectangle (background-agnostic)
      3. Conservative centre crop (last resort — always succeeds)
    """
    h, w = bgr.shape[:2]

    # --- Strategy 1: HSV background masks ---
    for mask_fn in [_mask_green_bg, _mask_white_bg, _mask_neutral_bg]:
        m = _clean_mask(mask_fn(bgr))
        ratio = cv2.countNonZero(m) / (h * w)
        if 0.03 <= ratio <= 0.85:
            if use_grabcut:
                refined = _mask_grabcut(bgr, m)
                refined = _clean_mask(refined)
                if cv2.countNonZero(refined) > 0:
                    m = refined
            bbox = _mask_to_bbox(m, h, w)
            if bbox is not None:
                return bbox, "hsv"

    # --- Strategy 2: GrabCut with centre rect ---
    if use_grabcut:
        gc = _grabcut_center_rect(bgr)
        if gc is not None:
            gc = _clean_mask(gc)
            bbox = _mask_to_bbox(gc, h, w)
            if bbox is not None:
                return bbox, "grabcut"

    # --- Strategy 3: centre-crop fallback (always succeeds) ---
    mx = max(1, int(w * 0.12))
    my = max(1, int(h * 0.12))
    fallback = np.zeros((h, w), dtype=np.uint8)
    fallback[my:h - my, mx:w - mx] = 255
    bbox = _mask_to_bbox(fallback, h, w, skip_quality=True)
    if bbox is not None:
        return bbox, "fallback"

    return None, "none"


MIN_BOX_PX = 60   # minimum box dimension in pixels


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def _draw_debug(bgr: np.ndarray, bbox: Optional[tuple]) -> np.ndarray:
    vis = bgr.copy()
    h, w = vis.shape[:2]
    if bbox:
        cx, cy, bw, bh = bbox
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "putter", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(vis, "NO DETECTION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return vis


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def label_images(
    images_dir: str,
    out_dir: str,
    review: bool = False,
    use_grabcut: bool = True,
    min_confidence: float = 0.0,   # reserved for future ML confidence
) -> dict:
    """
    Process all images in images_dir, generate YOLO labels,
    and copy accepted images to out_dir/images/all/.

    Returns stats dict.
    """
    src = Path(images_dir)
    dst = Path(out_dir)
    img_out  = dst / "images" / "all"
    lbl_out  = dst / "labels" / "all"
    rev_out  = dst / "review"

    for d in [img_out, lbl_out]:
        d.mkdir(parents=True, exist_ok=True)
    if review:
        rev_out.mkdir(parents=True, exist_ok=True)

    image_files = (
        list(src.glob("*.jpg")) +
        list(src.glob("*.jpeg")) +
        list(src.glob("*.png")) +
        list(src.glob("*.webp"))
    )
    image_files = [f for f in image_files if f.name != "_manifest.json"]

    print(f"[autolabel] v2 — 3-strategy detection (HSV → GrabCut → centre-crop)")
    print(f"[autolabel] Processing {len(image_files)} images…")

    stats = {"total": len(image_files), "labeled": 0, "failed": 0, "skipped": 0,
             "by_hsv": 0, "by_grabcut": 0, "by_fallback": 0}

    for img_path in tqdm(image_files, desc="Labeling", unit="img"):
        try:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                stats["skipped"] += 1
                continue

            # Resize very large images for speed
            h, w = bgr.shape[:2]
            if max(h, w) > 1200:
                scale = 1200 / max(h, w)
                bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)))

            bbox, strategy = detect_putter_bbox(bgr, use_grabcut=use_grabcut)

            if review:
                vis = _draw_debug(bgr, bbox)
                cv2.imwrite(str(rev_out / img_path.name), vis)

            if bbox is None or strategy == "none":
                stats["failed"] += 1
                continue

            # Track which strategy succeeded
            if strategy == "hsv":
                stats["by_hsv"] += 1
            elif strategy == "grabcut":
                stats["by_grabcut"] += 1
            elif strategy == "fallback":
                stats["by_fallback"] += 1

            # Save image + label
            out_stem = img_path.stem
            img_dest = img_out / (out_stem + ".jpg")
            lbl_dest = lbl_out / (out_stem + ".txt")

            # Re-read original (not resized) for the saved image
            original = cv2.imread(str(img_path))
            if original is None:
                original = bgr
            cv2.imwrite(str(img_dest), original, [cv2.IMWRITE_JPEG_QUALITY, 92])

            cx, cy, bw, bh = bbox
            with open(lbl_dest, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            stats["labeled"] += 1

        except Exception as e:
            print(f"  Error on {img_path.name}: {e}")
            stats["skipped"] += 1

    # Write data.yaml
    _write_yaml(dst)

    print(f"\n[autolabel] Results:")
    print(f"  Labeled:  {stats['labeled']}  "
          f"(HSV: {stats['by_hsv']}, GrabCut: {stats['by_grabcut']}, "
          f"Fallback: {stats['by_fallback']})")
    print(f"  Failed:   {stats['failed']}  (no putter detected)")
    print(f"  Skipped:  {stats['skipped']}  (corrupt / unreadable)")
    print(f"  Output:   {dst.resolve()}")
    if review:
        print(f"  Review:   {rev_out.resolve()}")
    print(f"\n  → Run next: python training/autolabel.py split --dataset {dst}")
    return stats


def _write_yaml(dataset_dir: Path) -> None:
    yaml = f"""# PutterTrack Pro – putter detection dataset
path: {dataset_dir.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: putter
"""
    with open(dataset_dir / "data.yaml", "w") as f:
        f.write(yaml)


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_dataset(
    dataset_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Split images/all + labels/all into train/val subsets.
    Moves (not copies) files.
    """
    dst = Path(dataset_dir)
    img_all  = dst / "images" / "all"
    lbl_all  = dst / "labels" / "all"

    if not img_all.exists():
        print(f"[split] images/all not found in {dataset_dir}. Run autolabel first.")
        return

    stems = [p.stem for p in img_all.glob("*.jpg")]
    random.seed(seed)
    random.shuffle(stems)

    n_val   = max(1, int(len(stems) * val_ratio))
    val_set = set(stems[:n_val])
    trn_set = set(stems[n_val:])

    for split_name, stem_set in [("train", trn_set), ("val", val_set)]:
        (dst / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        for stem in stem_set:
            for suffix, src_dir, dst_dir in [
                (".jpg",  img_all,  dst / "images" / split_name),
                (".txt",  lbl_all,  dst / "labels" / split_name),
            ]:
                src_file = src_dir / (stem + suffix)
                if src_file.exists():
                    shutil.move(str(src_file), str(dst_dir / (stem + suffix)))

    print(f"[split] Train: {len(trn_set)}  |  Val: {len(val_set)}")
    print(f"[split] Dataset ready at {dst.resolve()}")
    print(f"\n  → Train now:")
    print(f"    python training/train_detector.py train --data {dst}/data.yaml --epochs 100")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-label putter images for YOLO training"
    )
    sub = parser.add_subparsers(dest="cmd")

    # label command
    lp = sub.add_parser("label", help="Label images and build dataset structure")
    lp.add_argument("--images",  required=True, help="Raw image directory")
    lp.add_argument("--out",     default="data/putter_dataset", help="Output dataset dir")
    lp.add_argument("--review",  action="store_true",
                    help="Save debug visualisations to <out>/review/")
    lp.add_argument("--no-grabcut", action="store_true",
                    help="Skip GrabCut refinement (faster but less accurate)")

    # split command
    sp = sub.add_parser("split", help="Split labeled dataset into train/val")
    sp.add_argument("--dataset",   default="data/putter_dataset")
    sp.add_argument("--val-ratio", type=float, default=0.2)

    args = parser.parse_args()

    if args.cmd == "label":
        label_images(
            images_dir=args.images,
            out_dir=args.out,
            review=args.review,
            use_grabcut=not args.no_grabcut,
        )
    elif args.cmd == "split":
        split_dataset(args.dataset, val_ratio=args.val_ratio)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
