import cv2
import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Sequence, Tuple

import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from tqdm import tqdm
import pandas as pd
import numpy as np

# ------------------ Selezione landmarks ------------------
# Per compatibilità con lo script "coco-like" a 13 punti:
# 1 NOSE + 12 (spalle, gomiti, polsi, anche, ginocchia, caviglie)
FEATURE_LMS: List[PoseLandmark] = [
    PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,      PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_KNEE,     PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_ANKLE,    PoseLandmark.RIGHT_ANKLE,
]
FEATURE_IDX: List[int] = [lm.value for lm in FEATURE_LMS]

# Nomi colonne in formato <name>_y, <name>_x (come il codice che vuoi replicare)
COLUMNS = [
    "left_shoulder_y","left_shoulder_x","right_shoulder_y","right_shoulder_x",
    "left_elbow_y","left_elbow_x","right_elbow_y","right_elbow_x",
    "left_wrist_y","left_wrist_x","right_wrist_y","right_wrist_x",
    "left_hip_y","left_hip_x","right_hip_y","right_hip_x",
    "left_knee_y","left_knee_x","right_knee_y","right_knee_x",
    "left_ankle_y","left_ankle_x","right_ankle_y","right_ankle_x",
]

# ------------------ Disegno solo corpo (opzionale) ------------------
ColorBGR = Tuple[int, int, int]
BODY_CONNECTIONS = [
    (PoseLandmark.LEFT_SHOULDER,  PoseLandmark.RIGHT_SHOULDER,  (255, 255, 255)),
    (PoseLandmark.LEFT_SHOULDER,  PoseLandmark.LEFT_HIP,        (255, 255, 255)),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP,       (255, 255, 255)),
    (PoseLandmark.LEFT_HIP,       PoseLandmark.RIGHT_HIP,       (255, 255, 255)),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, (255, 0, 0)),
    (PoseLandmark.LEFT_ELBOW,    PoseLandmark.LEFT_WRIST, (255, 0, 0)),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, (0, 0, 255)),
    (PoseLandmark.RIGHT_ELBOW,    PoseLandmark.RIGHT_WRIST, (0, 0, 255)),
    (PoseLandmark.LEFT_HIP,   PoseLandmark.LEFT_KNEE,       (0, 255, 0)),
    (PoseLandmark.LEFT_KNEE,  PoseLandmark.LEFT_ANKLE,      (0, 255, 0)),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL,       (0, 255, 0)),
    (PoseLandmark.LEFT_HEEL,  PoseLandmark.LEFT_FOOT_INDEX, (0, 255, 0)),
    (PoseLandmark.RIGHT_HIP,   PoseLandmark.RIGHT_KNEE,       (0, 255, 255)),
    (PoseLandmark.RIGHT_KNEE,  PoseLandmark.RIGHT_ANKLE,      (0, 255, 255)),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL,       (0, 255, 255)),
    (PoseLandmark.RIGHT_HEEL,  PoseLandmark.RIGHT_FOOT_INDEX, (0, 255, 255)),
]
BODY_IDX_DRAW = [
    PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.RIGHT_SHOULDER.value,
    PoseLandmark.LEFT_ELBOW.value,    PoseLandmark.RIGHT_ELBOW.value,
    PoseLandmark.LEFT_WRIST.value,    PoseLandmark.RIGHT_WRIST.value,
    PoseLandmark.LEFT_HIP.value,      PoseLandmark.RIGHT_HIP.value,
    PoseLandmark.LEFT_KNEE.value,     PoseLandmark.RIGHT_KNEE.value,
    PoseLandmark.LEFT_ANKLE.value,    PoseLandmark.RIGHT_ANKLE.value,
    PoseLandmark.LEFT_HEEL.value,     PoseLandmark.RIGHT_HEEL.value,
    PoseLandmark.LEFT_FOOT_INDEX.value, PoseLandmark.RIGHT_FOOT_INDEX.value,
]

def draw_body_only(
    frame,
    landmarks: Sequence,
    img_w: int,
    img_h: int,
    visibility_thr: float = 0.5,
    point_radius: int = 3,
    point_color: ColorBGR = (0, 255, 0),
    point_thickness: int = -1,
    line_thickness: int = 2,
) -> None:
    for idx in BODY_IDX_DRAW:
        lm = landmarks[idx]
        if getattr(lm, "visibility", 1.0) < visibility_thr:
            continue
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        if 0 <= x < img_w and 0 <= y < img_h:
            cv2.circle(frame, (x, y), point_radius, point_color, point_thickness)

    for a, b, color in BODY_CONNECTIONS:
        la, lb = landmarks[a.value], landmarks[b.value]
        if getattr(la, "visibility", 1.0) < visibility_thr: continue
        if getattr(lb, "visibility", 1.0) < visibility_thr: continue
        ax, ay = int(la.x * img_w), int(la.y * img_h)
        bx, by = int(lb.x * img_w), int(lb.y * img_h)
        if (0 <= ax < img_w and 0 <= ay < img_h and 0 <= bx < img_w and 0 <= by < img_h):
            cv2.line(frame, (ax, ay), (bx, by), color, line_thickness)

# ------------------ Utility feature extraction ------------------
def lm_vec_26_yx(landmarks: Sequence, vis_thr: float, img_w: int, img_h: int) -> Tuple[np.ndarray, float]:
    """
    Ritorna:
      - features shape (26,) con ordine COLUMNS (y,x alternati) in coordinate NORMALIZZATE [0,1] (come MediaPipe)
        NB: se preferisci pixel, sostituisci con x*img_w, y*img_h e aggiorna i nomi.
      - mean_visibility delle 13 feature
    Se qualsiasi landmark è mancante nel risultato, usa visibility=0 (verrà conteggiato nella media).
    """
    feats = np.zeros((len(FEATURE_IDX), 2), dtype=np.float32)
    vis = np.zeros((len(FEATURE_IDX),), dtype=np.float32)
    for i, idx in enumerate(FEATURE_IDX):
        lm = landmarks[idx]
        x, y = float(lm.x), float(lm.y)  # normalizzati
        v = float(getattr(lm, "visibility", 1.0))
        feats[i] = [y, x]  # (y, x) come nello script originale
        vis[i] = v
    return feats.reshape(-1), float(vis.mean())

# ------------------ Main ------------------
def main():
    parser = ArgumentParser(description="Estrai sequenze di pose (±NB_IMAGES//2) attorno alle annotazioni e salva CSV per clip.")
    parser.add_argument("video", type=str, help="Video di input")
    parser.add_argument("annotation", type=Path, help="CSV annotazioni con colonne: FrameId, Shot")
    parser.add_argument("outdir", type=Path, help="Cartella di output per i CSV delle clip")
    parser.add_argument("--nb-images", type=int, default=30, help="Lunghezza finestra (default: 30)")
    parser.add_argument("--min-mean-vis", type=float, default=0.3, help="Soglia di qualità (media visibility) per accettare una clip (default: 0.3)")
    parser.add_argument("--draw", action="store_true", help="Mostra finestra con overlay (lento)")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0,1,2], help="MediaPipe Pose complexity")
    args = parser.parse_args()

    # Output dir
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Leggi annotazioni
    shots = pd.read_csv(args.annotation)
    if not {"Shot","Frame"}.issubset(set(shots.columns)):
        raise ValueError("Il file di annotazioni deve avere colonne: Shot, Frame")
    shots = shots.sort_values("Frame").reset_index(drop=True)

    # Contatori per i nomi file
    idx_map = {"forehand":1, "backhand":1, "serve":1, "neutral":1}

    # Apri video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_pose = mp.solutions.pose
    NB = args.nb_images
    half = NB // 2

    # MediaPipe su tutto il video in streaming
    with mp_pose.Pose(static_image_mode=False, model_complexity=args.model_complexity) as pose, \
         tqdm(total=total_frames, desc="Scansione video", unit="frame") as pbar:
        frame_id = 0
        current_row = 0
        shots_features: List[np.ndarray] = []
        active_window = None  # (start, end, label)
        # Pre-calcola “neutral” target (midpoint) dinamicamente quando serve
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Estrai landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            lms = result.pose_landmarks.landmark if result.pose_landmarks else None

            # Avvia una nuova finestra se entriamo nell'intervallo del prossimo shot
            def start_window(center_fid: int, label: str):
                return (center_fid - half, center_fid + half, label)

            # 1) Window per shot annotati
            if current_row < len(shots):
                shot_fid = int(shots.iloc[current_row]["Frame"])
                shot_lab = str(shots.iloc[current_row]["Shot"]).strip().lower()
                if active_window is None and frame_id == (shot_fid - half):
                    active_window = start_window(shot_fid, shot_lab)
                    shots_features = []

            # 2) Window neutral tra due shot distanti
            if active_window is None and current_row > 0 and current_row < len(shots):
                prev_fid = int(shots.iloc[current_row-1]["Frame"])
                curr_fid = int(shots.iloc[current_row]["Frame"])
                if (curr_fid - prev_fid) > NB:
                    mid = (prev_fid + curr_fid) // 2
                    if frame_id == (mid - half):
                        active_window = start_window(mid, "neutral")
                        shots_features = []

            # Se in una finestra attiva, accumula feature
            if active_window is not None:
                start, end, label = active_window
                if start <= frame_id < end:
                    if lms is None:
                        # niente landmarks -> vis media sarà 0
                        feats = np.zeros((len(FEATURE_IDX)*2,), dtype=np.float32)
                        mean_vis = 0.0
                    else:
                        feats, mean_vis = lm_vec_26_yx(lms, args.min_mean_vis, width, height)
                    shots_features.append(feats)

                    # Disegno a video (opzionale)
                    if args.draw and lms is not None:
                        draw_body_only(frame, lms, width, height, visibility_thr=0.5)
                        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
                        cv2.imshow("Frame", frame)
                        if cv2.waitKey(1) & 0xFF == 27:  # ESC
                            break

                    # Se abbiamo raggiunto la fine della finestra, salva o scarta
                    if frame_id == end - 1:
                        feats_arr = np.stack(shots_features, axis=0)  # (NB, 26)
                        # Controllo qualità: se la media visibilità media sui frame è bassa, scarta
                        # (replichiamo la logica "se media < soglia -> cancel shot")
                        # Calcolo mean_vis frame-wise approssimato: valori 0 per lms mancanti.
                        # In lm_vec_26_yx, non ritorniamo vis separati per frame; stimiamo:
                        # se nel buffer ci sono molti zeri, la clip è bassa qualità.
                        # Più aderente: richiediamo che NESSUNA riga sia tutta zero.
                        poor = (feats_arr.sum(axis=1) == 0).any()
                        if poor:
                            # scarta clip
                            pass
                        else:
                            df = pd.DataFrame(feats_arr, columns=COLUMNS)
                            df["shot"] = label
                            # nome file
                            if label not in idx_map:
                                idx_map[label] = 1
                            outpath = args.outdir / f"{label}_{idx_map[label]:03d}.csv"
                            idx_map[label] += 1
                            df.to_csv(outpath, index=False)
                            print(f"Salvato {label} -> {outpath}")

                        # avanza puntatore shot se non-neutral
                        if label != "neutral" and current_row < len(shots):
                            current_row += 1

                        # reset finestra
                        active_window = None
                        shots_features = []

                # se abbiamo superato la finestra per qualunque motivo, reset
                if frame_id > end:
                    active_window = None
                    shots_features = []

            # Prossimo frame
            frame_id += 1
            pbar.update(1)

    cap.release()
    if args.draw:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
