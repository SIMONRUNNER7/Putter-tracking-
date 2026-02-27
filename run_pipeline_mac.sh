#!/usr/bin/env bash
# ============================================================
#  PutterTrack Pro — Pipeline macOS
#  Usage : bash run_pipeline_mac.sh
#  Depuis la racine du projet
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

# ---------- couleurs ----------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; exit 1; }

echo ""
echo "============================================"
echo "  PutterTrack Pro — Dataset & Entraînement"
echo "  Mac Edition 🍎"
echo "============================================"

# ---------- 0. Python ----------
echo ""
echo "[0/5] Vérification de l'environnement…"

# Python 3.9+
PY=$(command -v python3 || command -v python || error "Python introuvable. Installe Python 3.9+ via https://brew.sh → 'brew install python'")
PY_VER=$("$PY" -c "import sys; print(sys.version_info >= (3,9))")
[ "$PY_VER" = "True" ] || error "Python 3.9+ requis. Version actuelle : $("$PY" --version)"
info "Python : $("$PY" --version)"

# pip
"$PY" -m pip install -q --upgrade pip

# Dépendances
info "Installation des dépendances…"
"$PY" -m pip install -q \
    duckduckgo-search \
    requests beautifulsoup4 \
    Pillow tqdm aiohttp \
    "ultralytics>=8.0" \
    "opencv-python-headless"

# ---------- Détection GPU (Apple Silicon vs Intel) ----------
DEVICE=$("$PY" -c "
import torch
if torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
")
if [ "$DEVICE" = "mps" ]; then
    info "Apple Silicon détecté → entraînement sur GPU Metal (MPS)"
else
    warn "Pas de Metal/MPS → entraînement sur CPU (plus lent)"
fi

# ---------- 1. Scraping ----------
echo ""
echo "[1/5] Scraping d'images de putters (cible : 300)…"
"$PY" training/scraper.py scrape \
    --out     data/raw_images \
    --limit   300 \
    --workers 6

"$PY" training/scraper.py stats --dir data/raw_images

# ---------- 2. Auto-labeling ----------
echo ""
echo "[2/5] Auto-labeling (HSV + GrabCut)…"
"$PY" training/autolabel.py label \
    --images data/raw_images \
    --out    data/putter_dataset \
    --review

# ---------- 3. Split train/val ----------
echo ""
echo "[3/5] Split 80% train / 20% val…"
"$PY" training/autolabel.py split \
    --dataset   data/putter_dataset \
    --val-ratio 0.2

# ---------- 4. Augmentation ----------
echo ""
echo "[4/5] Augmentation ×5…"
"$PY" training/augment.py \
    --dataset data/putter_dataset \
    --factor  5

# ---------- 5. Entraînement YOLOv8 ----------
echo ""
echo "[5/5] Entraînement YOLOv8 nano (device=$DEVICE)…"

# Batch plus petit sur MPS pour éviter les OOM
BATCH=8
[ "$DEVICE" = "mps" ] && BATCH=16

"$PY" training/train_detector.py train \
    --data   data/putter_dataset/data.yaml \
    --epochs 100 \
    --imgsz  640 \
    --batch  $BATCH \
    --model  yolov8n.pt \
    --output runs/putter \
    --device "$DEVICE"

# ---------- Résumé ----------
echo ""
echo "============================================"
echo "  Entraînement terminé !"
echo ""
echo "  Meilleurs poids :"
echo "  runs/putter/putter_detector/weights/best.pt"
echo ""
echo "  Pour utiliser dans PutterTrack Pro :"
echo "  Analyse → Utiliser modèle YOLO → best.pt"
echo "============================================"
