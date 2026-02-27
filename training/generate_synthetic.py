"""
Synthetic putter image generator.

Generates procedural putter-head images on green backgrounds
so you can test the full labeling + training pipeline without
internet access or a real image dataset.

Generated images are realistic enough for:
  - Testing the auto-labeler (HSV mask + GrabCut)
  - Testing the augmentation pipeline
  - Basic YOLO training sanity check

To build a real dataset, run scraper.py on a machine with internet
access and use those images instead of / in addition to these.

Usage:
    python training/generate_synthetic.py \
        --out   data/raw_images \
        --count 300

    # Then run the normal pipeline:
    python training/autolabel.py label --images data/raw_images --out data/putter_dataset
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Green background generators
# ---------------------------------------------------------------------------

def _gen_grass_bg(h: int, w: int) -> np.ndarray:
    """Generate a realistic grass/putting-green texture."""
    # Base green
    base = np.zeros((h, w, 3), dtype=np.float32)
    hue   = random.uniform(55, 85)       # green hue in degrees
    sat   = random.uniform(0.35, 0.75)
    val   = random.uniform(0.25, 0.65)

    # Convert HSV → BGR
    h_norm = hue / 360
    r, g, b = _hsv_to_rgb(h_norm, sat, val)
    base[:, :] = [b * 255, g * 255, r * 255]

    # Add grass stripe texture
    stripe_w = random.randint(12, 30)
    for x in range(0, w, stripe_w * 2):
        factor = random.uniform(0.85, 1.0)
        base[:, x:x+stripe_w] *= factor

    # Perlin-ish noise via blurred random
    noise = np.random.uniform(0.85, 1.15, (h // 8, w // 8)).astype(np.float32)
    noise = cv2.resize(noise, (w, h))
    base  = (base * noise[..., np.newaxis]).clip(0, 255).astype(np.uint8)

    return base


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple:
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


# ---------------------------------------------------------------------------
# Putter shape generators
# ---------------------------------------------------------------------------

def _draw_blade_putter(canvas: np.ndarray, cx: int, cy: int,
                       scale: float, color: tuple, angle: float = 0) -> np.ndarray:
    """Draw a blade-style putter head."""
    # Head: wide flat rectangle
    hw = int(90 * scale)
    hh = int(22 * scale)
    hosel_h = int(60 * scale)
    hosel_w = int(10 * scale)

    pts_head = np.array([
        [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
    ], dtype=np.float32)

    # Rotate
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts_head = (R @ pts_head.T).T + [cx, cy]
    pts_head = pts_head.astype(np.int32)

    cv2.fillPoly(canvas, [pts_head], color)

    # Face line (darker)
    face_color = tuple(max(0, c - 40) for c in color)
    face_start = (R @ np.array([-hw, 0])) + [cx, cy]
    face_end   = (R @ np.array([-hw, hh])) + [cx, cy]
    cv2.line(canvas, tuple(face_start.astype(int)), tuple(face_end.astype(int)),
             face_color, int(4 * scale))

    # Hosel (neck)
    hosel_top   = (R @ np.array([hw * 0.3, -hh])) + [cx, cy]
    hosel_bot   = (R @ np.array([hw * 0.3, -hh - hosel_h])) + [cx, cy]
    cv2.line(canvas, tuple(hosel_top.astype(int)), tuple(hosel_bot.astype(int)),
             color, hosel_w)

    return canvas


def _draw_mallet_putter(canvas: np.ndarray, cx: int, cy: int,
                        scale: float, color: tuple, angle: float = 0) -> np.ndarray:
    """Draw a mallet-style putter head (semi-circle / D-shape)."""
    head_r = int(55 * scale)
    head_w = int(120 * scale)
    head_h = int(50 * scale)
    hosel_h = int(55 * scale)
    hosel_w = int(10 * scale)

    # Ellipse for mallet body
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    axes = (head_w // 2, head_h // 2)
    angle_deg = int(np.degrees(angle))
    cv2.ellipse(canvas, (cx, cy), axes, angle_deg, 0, 360, color, -1)

    # Alignment aid (white line on top)
    aid_color = (220, 220, 220)
    p1 = (R @ np.array([0, -head_h // 2])) + [cx, cy]
    p2 = (R @ np.array([0, -head_h // 2 - int(20 * scale)])) + [cx, cy]
    cv2.line(canvas, tuple(p1.astype(int)), tuple(p2.astype(int)),
             aid_color, max(2, int(4 * scale)))

    # Hosel
    hosel_top = (R @ np.array([head_w * 0.25, -head_h // 2])) + [cx, cy]
    hosel_bot = (R @ np.array([head_w * 0.25, -head_h // 2 - hosel_h])) + [cx, cy]
    cv2.line(canvas, tuple(hosel_top.astype(int)), tuple(hosel_bot.astype(int)),
             color, hosel_w)

    return canvas


def _draw_spider_putter(canvas: np.ndarray, cx: int, cy: int,
                        scale: float, color: tuple, angle: float = 0) -> np.ndarray:
    """Draw a spider/frame-style mallet putter (TaylorMade look)."""
    hw = int(70 * scale)
    hh = int(45 * scale)
    frame_t = int(12 * scale)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    def rpt(x, y):
        return tuple((R @ np.array([x, y]) + [cx, cy]).astype(int))

    # Frame: 4 sides
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    for i in range(4):
        p1 = rpt(*corners[i])
        p2 = rpt(*corners[(i+1) % 4])
        cv2.line(canvas, p1, p2, color, frame_t)

    # Cross-bar (spider arm)
    cv2.line(canvas, rpt(-hw, 0), rpt(hw, 0), color, frame_t // 2)
    cv2.line(canvas, rpt(0, -hh), rpt(0, hh), color, frame_t // 2)

    # Insert (face, darker)
    insert_color = tuple(max(0, c - 50) for c in color)
    insert_pts = np.array([
        rpt(-hw, -hh), rpt(-hw + frame_t * 2, -hh),
        rpt(-hw + frame_t * 2, hh), rpt(-hw, hh),
    ])
    cv2.fillPoly(canvas, [insert_pts], insert_color)

    # Hosel
    hosel_top = rpt(hw * 0.3, -hh)
    hosel_bot = rpt(hw * 0.3, -hh - int(55 * scale))
    cv2.line(canvas, hosel_top, hosel_bot, color, int(10 * scale))

    return canvas


# ---------------------------------------------------------------------------
# Surface / finish
# ---------------------------------------------------------------------------

def _apply_metallic(canvas: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Add a subtle metallic sheen to the putter area."""
    h, w = canvas.shape[:2]
    # Gradient highlight
    grad = np.zeros((h, w), dtype=np.float32)
    # Diagonal highlight
    for y in range(h):
        for x in range(w):
            grad[y, x] = (x + y) / (h + w)
    grad = cv2.GaussianBlur(grad, (31, 31), 0)

    sheen = (grad * 30).astype(np.int16)
    fg = mask > 127
    canvas_float = canvas.astype(np.int16)
    canvas_float[fg, :] += sheen[fg, np.newaxis]
    return canvas_float.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

PUTTER_STYLES = ["blade", "mallet", "spider"]

# Steel / dark metal / gunmetal / raw / black
PUTTER_COLORS = [
    (140, 140, 145),   # steel grey
    (100, 100, 105),   # darker steel
    ( 70,  70,  75),   # gunmetal
    ( 55,  50,  50),   # dark
    (180, 175, 165),   # silver/chrome
    ( 85,  80,  75),   # raw/aged steel
    ( 40,  35,  35),   # matte black
    (115, 110, 100),   # satin finish
]


def generate_putter_image(
    h: int = 480,
    w: int = 640,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Generate one synthetic putter image on a grass background.
    Returns (bgr_image, yolo_bbox) where bbox = (cx, cy, bw, bh) normalised.
    """
    canvas = _gen_grass_bg(h, w)

    style  = random.choice(PUTTER_STYLES)
    color  = random.choice(PUTTER_COLORS)
    scale  = random.uniform(0.7, 1.4)
    angle  = random.uniform(-0.25, 0.25)   # ±~15°

    # Position: putter in lower-centre, with variation
    cx = int(w * random.uniform(0.3, 0.7))
    cy = int(h * random.uniform(0.4, 0.75))

    if style == "blade":
        _draw_blade_putter(canvas, cx, cy, scale, color, angle)
    elif style == "mallet":
        _draw_mallet_putter(canvas, cx, cy, scale, color, angle)
    else:
        _draw_spider_putter(canvas, cx, cy, scale, color, angle)

    # Create mask to compute real bbox
    diff = cv2.absdiff(canvas, _gen_grass_bg(h, w))
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Apply metallic finish
    canvas = _apply_metallic(canvas, mask)

    # Soft blur (simulate depth of field on real photos)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    # Bounding box
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        # Add padding
        pad = 15
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        norm_cx = (x1 + x2) / 2 / w
        norm_cy = (y1 + y2) / 2 / h
        norm_bw = (x2 - x1) / w
        norm_bh = (y2 - y1) / h
        bbox = (norm_cx, norm_cy, norm_bw, norm_bh)
    else:
        bbox = (0.5, 0.5, 0.5, 0.5)

    return canvas, bbox


def generate_dataset(out_dir: str, count: int = 300) -> None:
    """
    Generate `count` synthetic putter images + YOLO labels.
    Images are saved directly with labels so you can optionally
    skip the auto-labeler and go straight to augment + train.
    """
    out   = Path(out_dir)
    imgs  = out / "images" / "all"
    lbls  = out / "labels" / "all"
    raw   = out / "raw"     # also save to raw/ for autolabeler testing

    for d in [imgs, lbls, raw]:
        d.mkdir(parents=True, exist_ok=True)

    sizes = [(480, 640), (540, 720), (600, 800), (720, 960)]

    for i in tqdm(range(count), desc="Generating images", unit="img"):
        h, w  = random.choice(sizes)
        bgr, bbox = generate_putter_image(h, w)

        stem = f"synth_{i:04d}"

        # Save to raw/ (for autolabel testing)
        cv2.imwrite(str(raw / f"{stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Save pre-labeled version
        cv2.imwrite(str(imgs / f"{stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cx, cy, bw, bh = bbox
        with open(lbls / f"{stem}.txt", "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # Write data.yaml
    yaml = f"""# PutterTrack Pro — synthetic dataset
path: {out.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: putter
"""
    with open(out / "data.yaml", "w") as f:
        f.write(yaml)

    print(f"\n[synth] Generated {count} images → {out.resolve()}")
    print(f"\n  Pre-labeled dataset (ready to split):")
    print(f"    python training/autolabel.py split --dataset {out}")
    print(f"\n  Or test auto-labeler on raw images:")
    print(f"    python training/autolabel.py label --images {raw} --out {out}_labeled")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic putter images for pipeline testing"
    )
    parser.add_argument("--out",   default="data/putter_dataset",
                        help="Output dataset directory")
    parser.add_argument("--count", type=int, default=300,
                        help="Number of images to generate (default: 300)")
    args = parser.parse_args()
    generate_dataset(args.out, args.count)


if __name__ == "__main__":
    main()
