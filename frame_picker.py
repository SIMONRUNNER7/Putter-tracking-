#!/usr/bin/env python3
"""
frame_picker.py — Sélection manuelle des meilleures frames par colonne.

Usage:
    python3 frame_picker.py                  # premier shot dans captures/raw/
    python3 frame_picker.py captures/raw/shot_20240305_123456/

Interface:
    Clic sur le composite         → changer la colonne active
    Clic sur un thumbnail         → sélectionner cette frame
    Boutons en bas de la fenêtre  → toutes les actions disponibles
"""

import cv2
import numpy as np
import os
import sys
import json
import glob
import argparse
from pathlib import Path

N_COLS      = 7
BALL_CROP_R = 46      # rayon crop balle
THUMB_H     = 190     # hauteur zone thumbnails
THUMB_W     = 120     # largeur fixe d'un thumbnail (strip scrollable)
STATUS_H    = 28
BTN_H       = 50      # hauteur barre de boutons
WIN_NAME    = "Frame Picker"


# ── Export pour entraînement ─────────────────────────────────────────────────

def export_for_training(shot_dir, all_frames, selections, meta):
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from training.autolabel import detect_putter_bbox
        has_autolabel = True
    except ImportError:
        has_autolabel = False
        print("[export] autolabel non disponible → labels approchés (centre col)")

    ann_img = Path("annotation_output/images")
    ann_lbl = Path("annotation_output/labels")
    ann_img.mkdir(parents=True, exist_ok=True)
    ann_lbl.mkdir(parents=True, exist_ok=True)

    tag      = meta.get("tag", os.path.basename(shot_dir))
    n_all    = len(all_frames)
    exported = 0
    skipped  = 0

    for col_idx in range(N_COLS):
        sel = selections[col_idx]
        if n_all == 0 or sel >= n_all:
            skipped += 1
            continue

        frame = all_frames[sel]
        stem  = f"{tag}_col{col_idx + 1}"

        img_path = ann_img / f"{stem}.jpg"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        bbox = None
        if has_autolabel:
            bbox, _ = detect_putter_bbox(frame)

        if bbox is None:
            h_img, w_img = frame.shape[:2]
            cx_n = (col_idx + 0.5) / N_COLS
            cy_n = 0.5
            bw_n = 1.0 / N_COLS
            bh_n = 80 / h_img
            bbox = (cx_n, cy_n, bw_n, bh_n)

        lbl_path = ann_lbl / f"{stem}.txt"
        cx, cy, bw, bh = bbox
        with open(lbl_path, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        exported += 1

    total = exported + skipped
    print(f"[export] {exported}/{total} cols → annotation_output/")
    if skipped:
        print(f"[export] {skipped} col(s) vide(s) ignorée(s)")
    return exported


# ── Lecture des shots ─────────────────────────────────────────────────────────

def find_all_shots(base="captures/raw"):
    return sorted(glob.glob(os.path.join(base, "shot_*")))


def find_latest_shot(base="captures/raw"):
    shots = find_all_shots()
    return shots[-1] if shots else None


def load_shot(shot_dir):
    meta_path = os.path.join(shot_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"[erreur] metadata.json introuvable dans {shot_dir}")
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)

    # Nouveau format : pool global de toutes les frames d'enregistrement
    all_frames = []
    af_paths = sorted(glob.glob(os.path.join(shot_dir, "all_frames", "frame_*.jpg")))
    for p in af_paths:
        img = cv2.imread(p)
        if img is not None:
            all_frames.append(img)

    # Rétro-compat : pas de all_frames/ → aplatir les candidats par colonne
    if not all_frames:
        for col_idx in range(N_COLS):
            paths = sorted(glob.glob(
                os.path.join(shot_dir, f"col{col_idx + 1}", "frame_*.jpg")))
            for p in paths:
                img = cv2.imread(p)
                if img is not None:
                    all_frames.append(img)

    # Balle au repos
    ball_rest_kf = None
    p = os.path.join(shot_dir, "ball_rest.jpg")
    if os.path.exists(p):
        ball_rest_kf = cv2.imread(p)

    # Candidats balle col1
    ball1_frames = []
    ball1_pos    = meta.get("ball_col1_cands_pos", [])
    bc1_paths = sorted(glob.glob(
        os.path.join(shot_dir, "ball_col1_cands", "frame_*.jpg")))
    for p in bc1_paths:
        img = cv2.imread(p)
        if img is not None:
            ball1_frames.append(img)

    if not ball1_frames:
        p = os.path.join(shot_dir, "ball_col1.jpg")
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                ball1_frames = [img]
                if meta.get("ball_col_pos_0") is not None:
                    ball1_pos = [meta["ball_col_pos_0"]]

    return meta, all_frames, ball_rest_kf, ball1_frames, ball1_pos


def load_shot_state(shot_dir):
    meta, all_frames, ball_rest_kf, ball1_frames, ball1_pos = load_shot(shot_dir)
    n_all = len(all_frames)

    stored = meta.get("selected", {})
    selections = []
    for i in range(N_COLS):
        v = stored.get(str(i), 0)
        selections.append(min(v, n_all - 1) if n_all > 0 else 0)

    sel_ball1 = meta.get("sel_ball1", 0)
    if ball1_frames:
        sel_ball1 = min(sel_ball1, len(ball1_frames) - 1)

    return meta, all_frames, ball_rest_kf, ball1_frames, ball1_pos, selections, sel_ball1


# ── Composite ────────────────────────────────────────────────────────────────

def _paste_ball(canvas, kf_halfres, pos_halfres, W, H):
    if kf_halfres is None or pos_halfres is None:
        return
    r      = BALL_CROP_R
    kh, kw = kf_halfres.shape[:2]
    frame  = cv2.resize(kf_halfres, (W, H))
    sx = W / kw;  sy = H / kh
    cx = int(pos_halfres[0] * sx)
    cy = int(pos_halfres[1] * sy)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    yy, xx  = np.mgrid[y0:y1, x0:x1]
    dist    = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(np.float32)
    alpha   = np.clip(1.0 - (dist / r)**2, 0.0, 1.0)[:, :, np.newaxis]
    roi_f   = frame[y0:y1, x0:x1].astype(np.float32)
    roi_c   = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (roi_f * alpha + roi_c * (1 - alpha)).astype(np.uint8)


def build_composite(all_frames, selections, meta,
                    ball_rest_kf, ball1_frames, ball1_pos, sel_ball1,
                    active_col, ball1_mode):
    W, H   = meta["W"], meta["H"]
    col_w  = W // N_COLS
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    n_all  = len(all_frames)

    for col_idx in range(N_COLS):
        x0  = col_idx * col_w
        x1  = W if col_idx == N_COLS - 1 else x0 + col_w
        sel = selections[col_idx]
        if n_all > 0 and 0 <= sel < n_all:
            full = cv2.resize(all_frames[sel], (W, H))
            canvas[:, x0:x1] = full[:, x0:x1]

    for col_idx in range(1, N_COLS):
        cv2.line(canvas, (col_idx * col_w, 0), (col_idx * col_w, H),
                 (60, 60, 60), 1)
    for col_idx in range(N_COLS):
        cv2.putText(canvas, str(col_idx + 1),
                    (col_idx * col_w + 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    if not ball1_mode:
        ax0 = active_col * col_w
        ax1 = W if active_col == N_COLS - 1 else ax0 + col_w
        hl  = canvas.copy()
        cv2.rectangle(hl, (ax0, 0), (ax1, H), (0, 200, 255), -1)
        cv2.addWeighted(canvas, 0.82, hl, 0.18, 0, canvas)
        cv2.rectangle(canvas, (ax0, 0), (ax1, H), (0, 200, 255), 2)

    _paste_ball(canvas, ball_rest_kf, meta.get("ball_rest_pos"), W, H)

    if ball1_frames and 0 <= sel_ball1 < len(ball1_frames):
        pos = (ball1_pos[sel_ball1] if sel_ball1 < len(ball1_pos)
               else meta.get("ball_col_pos_0"))
        _paste_ball(canvas, ball1_frames[sel_ball1], pos, W, H)

    if ball1_mode:
        hl2 = canvas.copy()
        cv2.rectangle(hl2, (0, 0), (col_w, H), (0, 140, 255), -1)
        cv2.addWeighted(canvas, 0.82, hl2, 0.18, 0, canvas)
        cv2.rectangle(canvas, (0, 0), (col_w, H), (0, 140, 255), 2)
        cv2.putText(canvas, "BALLE", (6, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)

    return canvas


# ── Barre de boutons ──────────────────────────────────────────────────────────

def _draw_btn(bar, x0, x1, label, bg_color, text_color=(230, 230, 230),
              border_color=(160, 160, 160)):
    """Dessine un bouton dans la barre."""
    cv2.rectangle(bar, (x0 + 2, 2), (x1 - 2, BTN_H - 2), bg_color, -1)
    cv2.rectangle(bar, (x0 + 2, 2), (x1 - 2, BTN_H - 2), border_color, 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    tx = x0 + max(4, (x1 - x0 - tw) // 2)
    ty = BTN_H // 2 + th // 2
    cv2.putText(bar, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                text_color, 1, cv2.LINE_AA)


def build_buttons(W, shot_idx, n_shots, ball1_mode, dirty, n_frames,
                  active_col_idx, ball1_frames):
    """
    Construit la barre de boutons et retourne (image, regions).
    regions = liste de (action_id, x0, x1)
    """
    bar = np.full((BTN_H, W, 3), (30, 30, 30), dtype=np.uint8)

    # Définition des boutons dans l'ordre d'affichage
    # (label, action_id, bg_color, condition)
    btn_defs = []

    if n_shots > 1:
        btn_defs.append(("< SHOT",  "prev_shot",    (70, 70, 70)))
        btn_defs.append(("SHOT >",  "next_shot",    (70, 70, 70)))

    btn_defs.append(("< FRAME",    "prev_frame",   (60, 80, 110)))
    btn_defs.append(("FRAME >",    "next_frame",   (60, 80, 110)))

    ball_bg = (20, 110, 140) if ball1_mode else (20, 70, 90)
    btn_defs.append(("BALLE",      "toggle_ball",  ball_bg))

    save_bg = (20, 130, 20) if dirty else (20, 80, 20)
    btn_defs.append(("SAUVEGARDER","save",          save_bg))

    btn_defs.append(("EXPORTER",   "export",       (80, 60, 20)))
    btn_defs.append(("QUITTER",    "quit",         (100, 20, 20)))

    n_btns = len(btn_defs)
    bw     = W // n_btns
    regions = []

    for i, (label, action_id, bg) in enumerate(btn_defs):
        x0 = i * bw
        x1 = W if i == n_btns - 1 else x0 + bw
        _draw_btn(bar, x0, x1, label, bg)
        regions.append((action_id, x0, x1))

    # Indicateur shot en haut à droite
    if n_shots > 1:
        info = f"Shot {shot_idx + 1}/{n_shots}"
        cv2.putText(bar, info, (W - 140, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    # Indicateur frames en haut à gauche
    info_f = f"{n_frames} frames"
    cv2.putText(bar, info_f, (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1)

    return bar, regions


# ── Bande de thumbnails ───────────────────────────────────────────────────────

def build_strip(all_frames, sel, W,
                ball1_mode=False, ball1_frames=None, sel_ball1=0):
    if ball1_mode and ball1_frames:
        frames    = ball1_frames
        cur_sel   = sel_ball1
        bg        = (20, 10, 0)
        sel_color = (0, 140, 255)
        label_pfx = "B"
        tw        = min(W // len(frames), 260)
    else:
        frames    = all_frames
        cur_sel   = sel
        bg        = (20, 20, 20)
        sel_color = (0, 220, 0)
        label_pfx = ""
        tw        = THUMB_W

    strip = np.full((THUMB_H, W, 3), bg, dtype=np.uint8)

    if not frames:
        msg = "Aucun candidat balle" if ball1_mode else "Aucune frame enregistree"
        cv2.putText(strip, msg, (10, THUMB_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
        return strip

    n       = len(frames)
    max_vis = max(1, W // tw)
    half    = max_vis // 2
    start   = max(0, min(cur_sel - half, n - max_vis))
    end     = min(n, start + max_vis)

    for i, fi in enumerate(range(start, end)):
        x0 = i * tw
        if x0 >= W:
            break
        x1 = min(x0 + tw, W)
        th = cv2.resize(frames[fi], (x1 - x0, THUMB_H - 22))
        strip[0:THUMB_H - 22, x0:x1] = th

        label = f"{label_pfx}{fi + 1}"
        cv2.putText(strip, label, (x0 + 4, THUMB_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

        is_sel    = (fi == cur_sel)
        color     = sel_color if is_sel else (50, 50, 50)
        thickness = 3          if is_sel else 1
        cv2.rectangle(strip, (x0, 0), (x0 + tw - 1, THUMB_H - 23),
                      color, thickness)

    if not ball1_mode and n > max_vis:
        bar_y  = THUMB_H - 4
        bar_x0 = int(W * start / n)
        bar_x1 = int(W * end   / n)
        cv2.line(strip, (0, bar_y), (W, bar_y), (50, 50, 50), 2)
        cv2.line(strip, (bar_x0, bar_y), (bar_x1, bar_y), (100, 100, 100), 2)

    return strip


# ── Barre de statut (info seulement) ─────────────────────────────────────────

def build_status(all_frames, selections, active_col, dirty, W,
                 ball1_mode=False, ball1_frames=None, sel_ball1=0):
    n_all = len(all_frames)

    if ball1_mode and ball1_frames:
        n         = len(ball1_frames)
        sel       = sel_ball1 + 1
        col_label = "BALLE col1"
    else:
        n         = n_all
        sel       = (selections[active_col] + 1) if n_all > 0 else 0
        col_label = f"Col {active_col + 1}"

    state_txt = "NON SAUVEGARDE" if dirty else "sauvegarde"
    msg = f"  {col_label}  |  Frame {sel}/{n}  |  {state_txt}"
    bar = np.full((STATUS_H, W, 3), 15, dtype=np.uint8)
    cv2.putText(bar, msg, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (200, 160, 60) if dirty else (140, 200, 140), 1)
    return bar


# ── Rendu complet ─────────────────────────────────────────────────────────────

def render(all_frames, selections, meta, ball_rest_kf,
           ball1_frames, ball1_pos, sel_ball1,
           active_col, dirty, ball1_mode,
           shot_idx=0, n_shots=1, btn_regions_out=None):
    W   = meta["W"]
    H   = meta["H"]
    sel = selections[active_col] if active_col < N_COLS else 0

    composite = build_composite(all_frames, selections, meta,
                                ball_rest_kf, ball1_frames, ball1_pos,
                                sel_ball1, active_col, ball1_mode)

    btn_bar, regions = build_buttons(
        W, shot_idx, n_shots, ball1_mode, dirty,
        n_frames=len(all_frames),
        active_col_idx=active_col,
        ball1_frames=ball1_frames)

    if btn_regions_out is not None:
        btn_regions_out.clear()
        btn_regions_out.extend(regions)

    strip = cv2.resize(
        build_strip(all_frames, sel, W,
                    ball1_mode=ball1_mode,
                    ball1_frames=ball1_frames, sel_ball1=sel_ball1),
        (W, THUMB_H))

    status = build_status(all_frames, selections, active_col, dirty, W,
                          ball1_mode=ball1_mode, ball1_frames=ball1_frames,
                          sel_ball1=sel_ball1)

    return np.vstack([composite, btn_bar, strip, status])


# ── Sauvegarde ────────────────────────────────────────────────────────────────

def save(shot_dir, all_frames, selections, meta,
         ball_rest_kf, ball1_frames, ball1_pos, sel_ball1):
    W, H = meta["W"], meta["H"]

    meta["selected"]  = {str(k): v for k, v in enumerate(selections)}
    meta["sel_ball1"] = sel_ball1
    with open(os.path.join(shot_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    col_w  = W // N_COLS
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    n_all  = len(all_frames)
    for col_idx in range(N_COLS):
        x0  = col_idx * col_w
        x1  = W if col_idx == N_COLS - 1 else x0 + col_w
        sel = selections[col_idx]
        if n_all > 0 and 0 <= sel < n_all:
            full = cv2.resize(all_frames[sel], (W, H))
            canvas[:, x0:x1] = full[:, x0:x1]
    for col_idx in range(1, N_COLS):
        cv2.line(canvas, (col_idx * col_w, 0), (col_idx * col_w, H),
                 (60, 60, 60), 1)
    for col_idx in range(N_COLS):
        cv2.putText(canvas, str(col_idx + 1),
                    (col_idx * col_w + 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    _paste_ball(canvas, ball_rest_kf, meta.get("ball_rest_pos"), W, H)
    if ball1_frames and 0 <= sel_ball1 < len(ball1_frames):
        pos = (ball1_pos[sel_ball1] if sel_ball1 < len(ball1_pos)
               else meta.get("ball_col_pos_0"))
        _paste_ball(canvas, ball1_frames[sel_ball1], pos, W, H)

    comp_path = os.path.join(shot_dir, "composite.jpg")
    cv2.imwrite(comp_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

    tag = meta.get("tag", "")
    if tag:
        out_path = os.path.join("captures", f"strobe_{tag}_pick.png")
        cv2.imwrite(out_path, canvas)
        print(f"[save] composite → {out_path}")

    print(f"[save] metadata → {os.path.join(shot_dir, 'metadata.json')}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sélection manuelle des frames")
    parser.add_argument("shot_dir", nargs="?",
                        help="Dossier shot (captures/raw/shot_...)")
    args = parser.parse_args()

    all_shots = find_all_shots()
    n_shots   = len(all_shots)

    if args.shot_dir:
        shot_dir  = args.shot_dir
        shot_idx  = all_shots.index(shot_dir) if shot_dir in all_shots else 0
    else:
        shot_dir = all_shots[0] if all_shots else None
        shot_idx = 0

    if not shot_dir or not os.path.isdir(shot_dir):
        print("Aucun shot trouvé. Lance d'abord strobe_capture.py.")
        sys.exit(1)

    print(f"[picker] {n_shots} shot(s) | ouverture : {shot_dir}")

    (meta, all_frames, ball_rest_kf, ball1_frames, ball1_pos,
     selections, sel_ball1) = load_shot_state(shot_dir)
    print(f"[picker] {len(all_frames)} frame(s) chargées | "
          f"{len(ball1_frames)} frame(s) balle")
    W, H = meta["W"], meta["H"]

    active_col  = 3
    ball1_mode  = False
    dirty       = False
    btn_regions = []   # mis à jour à chaque render()

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, W, H + BTN_H + THUMB_H + STATUS_H)

    cb_state = {
        "active_col": active_col,
        "ball1_mode": False,
        "action":     None,   # action déclenchée par un bouton
    }

    def on_mouse(event, x, y, flags, param):
        nonlocal dirty, sel_ball1
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Zone composite → changer colonne active
        if y < H:
            col_w   = W // N_COLS
            clicked = min(x // col_w, N_COLS - 1)
            param["active_col"] = clicked
            param["ball1_mode"] = False
            return

        # Zone boutons
        if H <= y < H + BTN_H:
            for action_id, bx0, bx1 in btn_regions:
                if bx0 <= x < bx1:
                    param["action"] = action_id
            return

        # Zone strip thumbnails
        if H + BTN_H <= y < H + BTN_H + THUMB_H:
            if param["ball1_mode"] and ball1_frames:
                n  = len(ball1_frames)
                tw = min(W // n, 260)
                sel_ball1 = min(x // tw, n - 1)
                dirty = True
            else:
                n_all = len(all_frames)
                if not n_all:
                    return
                ac = param["active_col"]
                if ac >= N_COLS:
                    return
                max_vis = max(1, W // THUMB_W)
                cur_sel = selections[ac]
                half    = max_vis // 2
                start   = max(0, min(cur_sel - half, n_all - max_vis))
                clicked_idx = start + (x // THUMB_W)
                if 0 <= clicked_idx < n_all:
                    selections[ac] = clicked_idx
                    dirty = True

    cv2.setMouseCallback(WIN_NAME, on_mouse, cb_state)

    def switch_shot(new_idx):
        nonlocal shot_dir, shot_idx, meta, all_frames, ball_rest_kf
        nonlocal ball1_frames, ball1_pos, selections, sel_ball1
        nonlocal active_col, dirty, W, H
        if dirty:
            save(shot_dir, all_frames, selections, meta,
                 ball_rest_kf, ball1_frames, ball1_pos, sel_ball1)
            dirty = False
        shot_idx = max(0, min(new_idx, n_shots - 1))
        shot_dir = all_shots[shot_idx]
        print(f"[picker] Shot {shot_idx + 1}/{n_shots} : {shot_dir}")
        (meta, all_frames, ball_rest_kf, ball1_frames, ball1_pos,
         selections, sel_ball1) = load_shot_state(shot_dir)
        print(f"[picker] {len(all_frames)} frame(s) chargées | "
              f"{len(ball1_frames)} frame(s) balle")
        W, H = meta["W"], meta["H"]
        active_col = 3
        cb_state["active_col"] = 3
        cb_state["ball1_mode"] = False
        cv2.resizeWindow(WIN_NAME, W, H + BTN_H + THUMB_H + STATUS_H)

    while True:
        active_col = cb_state["active_col"]
        ball1_mode = cb_state["ball1_mode"]

        frame = render(all_frames, selections, meta, ball_rest_kf,
                       ball1_frames, ball1_pos, sel_ball1,
                       active_col, dirty, ball1_mode,
                       shot_idx=shot_idx, n_shots=n_shots,
                       btn_regions_out=btn_regions)
        cv2.imshow(WIN_NAME, frame)

        # ── Traitement des actions boutons ────────────────────────────────────
        action = cb_state.pop("action", None)
        if action is None:
            cb_state["action"] = None   # remettre la clé

        if action == "prev_shot":
            if shot_idx > 0:
                switch_shot(shot_idx - 1)
            else:
                print("[picker] Premier shot.")

        elif action == "next_shot":
            if shot_idx < n_shots - 1:
                switch_shot(shot_idx + 1)
            else:
                print("[picker] Dernier shot.")

        elif action == "prev_frame":
            if ball1_mode and ball1_frames:
                sel_ball1 = max(0, sel_ball1 - 1)
                dirty = True
            elif active_col < N_COLS and all_frames:
                selections[active_col] = max(0, selections[active_col] - 1)
                dirty = True

        elif action == "next_frame":
            if ball1_mode and ball1_frames:
                sel_ball1 = min(len(ball1_frames) - 1, sel_ball1 + 1)
                dirty = True
            elif active_col < N_COLS and all_frames:
                selections[active_col] = min(
                    len(all_frames) - 1, selections[active_col] + 1)
                dirty = True

        elif action == "toggle_ball":
            cb_state["ball1_mode"] = not ball1_mode
            print(f"[picker] Mode balle : {'ON' if not ball1_mode else 'OFF'}"
                  f" | {len(ball1_frames)} frame(s) balle")

        elif action == "save":
            save(shot_dir, all_frames, selections, meta,
                 ball_rest_kf, ball1_frames, ball1_pos, sel_ball1)
            dirty = False
            if shot_idx < n_shots - 1:
                switch_shot(shot_idx + 1)
            else:
                print("[picker] Dernier shot, tous sauvegardés.")

        elif action == "export":
            export_for_training(shot_dir, all_frames, selections, meta)

        elif action == "quit":
            break

        # ── Raccourcis clavier (conservés en complément) ──────────────────────
        key = cv2.waitKey(40) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key == ord('s'):
            save(shot_dir, all_frames, selections, meta,
                 ball_rest_kf, ball1_frames, ball1_pos, sel_ball1)
            dirty = False
            if shot_idx < n_shots - 1:
                switch_shot(shot_idx + 1)
            else:
                print("[picker] Dernier shot, tous sauvegardés.")

        elif key == ord('e'):
            export_for_training(shot_dir, all_frames, selections, meta)

        elif key in (ord('8'), ord('b')):
            cb_state["ball1_mode"] = not cb_state["ball1_mode"]

        elif key in range(ord('1'), ord('1') + N_COLS):
            cb_state["active_col"] = key - ord('1')
            cb_state["ball1_mode"] = False

        elif key in (ord('p'), ord('['), 44):
            if shot_idx > 0:
                switch_shot(shot_idx - 1)

        elif key in (ord('n'), ord(']'), 46):
            if shot_idx < n_shots - 1:
                switch_shot(shot_idx + 1)

        elif key in (81, ord(','), ord('a')):
            if ball1_mode and ball1_frames:
                sel_ball1 = max(0, sel_ball1 - 1)
                dirty = True
            elif active_col < N_COLS and all_frames:
                selections[active_col] = max(0, selections[active_col] - 1)
                dirty = True

        elif key in (83, ord('.'), ord('d')):
            if ball1_mode and ball1_frames:
                sel_ball1 = min(len(ball1_frames) - 1, sel_ball1 + 1)
                dirty = True
            elif active_col < N_COLS and all_frames:
                selections[active_col] = min(
                    len(all_frames) - 1, selections[active_col] + 1)
                dirty = True

    cv2.destroyAllWindows()
    if dirty:
        print("[picker] Modifications non sauvegardées.")


if __name__ == "__main__":
    main()
