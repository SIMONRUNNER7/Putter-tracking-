#!/usr/bin/env bash
# ============================================================
# PutterTrack Pro — Full training pipeline
# Run this from the project root: bash training/pipeline.sh
# ============================================================

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "  PutterTrack Pro — Dataset & Training"
echo "============================================"

# ---- 0. Dependencies
echo ""
echo "[0/5] Installing dependencies…"
pip install -q duckduckgo-search requests beautifulsoup4 Pillow tqdm aiohttp ultralytics opencv-python

# ---- 1. Scrape images
echo ""
echo "[1/5] Scraping putter images (300 target)…"
python training/scraper.py scrape \
    --out   data/raw_images \
    --limit 300 \
    --workers 6

python training/scraper.py stats --dir data/raw_images

# ---- 2. Auto-label
echo ""
echo "[2/5] Auto-labeling images (HSV + GrabCut)…"
python training/autolabel.py label \
    --images data/raw_images \
    --out    data/putter_dataset \
    --review

# ---- 3. Train/val split
echo ""
echo "[3/5] Splitting dataset (80% train / 20% val)…"
python training/autolabel.py split \
    --dataset   data/putter_dataset \
    --val-ratio 0.2

# ---- 4. Augmentation
echo ""
echo "[4/5] Augmenting training set (×5)…"
python training/augment.py \
    --dataset data/putter_dataset \
    --factor  5

# ---- 5. Train YOLOv8
echo ""
echo "[5/5] Training YOLOv8 nano model…"
echo "      (use --device 0 if you have a GPU, or 'mps' for Apple Silicon)"
python training/train_detector.py train \
    --data    data/putter_dataset/data.yaml \
    --epochs  100 \
    --imgsz   640 \
    --batch   16 \
    --model   yolov8n.pt \
    --output  runs/putter \
    --device  cpu

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Best weights: runs/putter/putter_detector/weights/best.pt"
echo ""
echo "  Load in PutterTrack Pro:"
echo "  Analysis → Use YOLO Model → select best.pt"
echo "============================================"
