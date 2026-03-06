#!/usr/bin/env python3
"""
export_all.py — Exporte toutes les frames sélectionnées de tous les shots.

Usage:
    python3 export_all.py

Lit tous les shots dans captures/raw/shot_*/
et génère annotation_output/images/ + annotation_output/labels/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frame_picker import (
    find_all_shots, load_shot_state, export_for_training
)


def main():
    shots = find_all_shots()
    if not shots:
        print("[export_all] Aucun shot trouvé dans captures/raw/")
        sys.exit(1)

    print(f"[export_all] {len(shots)} shot(s) trouvé(s)")
    total = 0

    for i, shot_dir in enumerate(shots):
        print(f"\n[export_all] Shot {i+1}/{len(shots)} : {shot_dir}")
        try:
            meta, all_frames, _, ball1_frames, ball1_pos, selections, sel_ball1 = \
                load_shot_state(shot_dir)
            n = export_for_training(shot_dir, all_frames, selections, meta)
            total += n
        except Exception as e:
            print(f"  [erreur] {e}")

    print(f"\n[export_all] Total exporté : {total} images → annotation_output/")


if __name__ == "__main__":
    main()
