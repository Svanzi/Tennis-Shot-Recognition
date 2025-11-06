import cv2
import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Sequence, Tuple
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from tqdm import tqdm
import numpy as np
from collections import deque
import tensorflow as tf
import os

##########################################################################################################
# Script to annotate a video with body-only landmarks using MediaPipe Pose,
# and export a CSV file with the landmark coordinates per frame.
# Usage:
#   python Landmarker.py input_video.mp4
# Outputs:
#   - Annotated video in ./Landmarked/input_video_Landmarked.mp4 
#   - CSV file with landmark coordinates in ./Landmarked/input_video_Landmarked.csv
##########################################################################################################


# --------- Constants & Types ---------
# OpenCV uses BGR color order (NOT RGB).
ColorBGR = Tuple[int, int, int]
Connection = Tuple[PoseLandmark, PoseLandmark, ColorBGR]

# Body landmark indices (no face)
BODY_LANDMARKS: List[PoseLandmark] = [
    PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,      PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_KNEE,     PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_ANKLE,    PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.LEFT_HEEL,     PoseLandmark.RIGHT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX, PoseLandmark.RIGHT_FOOT_INDEX,
]
BODY_IDX: List[int] = [lm.value for lm in BODY_LANDMARKS]

# Body-only connections (no face), with OpenCV BGR colors
BODY_CONNECTIONS: List[Connection] = [
    # Torso / shoulders / hips -> white
    (PoseLandmark.LEFT_SHOULDER,  PoseLandmark.RIGHT_SHOULDER,  (255, 255, 255)),
    (PoseLandmark.LEFT_SHOULDER,  PoseLandmark.LEFT_HIP,        (255, 255, 255)),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP,       (255, 255, 255)),
    (PoseLandmark.LEFT_HIP,       PoseLandmark.RIGHT_HIP,       (255, 255, 255)),

    # Left arm -> blue
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, (255, 0, 0)),
    (PoseLandmark.LEFT_ELBOW,    PoseLandmark.LEFT_WRIST, (255, 0, 0)),

    # Right arm -> red
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, (0, 0, 255)),
    (PoseLandmark.RIGHT_ELBOW,    PoseLandmark.RIGHT_WRIST, (0, 0, 255)),

    # Left leg -> green
    (PoseLandmark.LEFT_HIP,   PoseLandmark.LEFT_KNEE,       (0, 255, 0)),
    (PoseLandmark.LEFT_KNEE,  PoseLandmark.LEFT_ANKLE,      (0, 255, 0)),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL,       (0, 255, 0)),
    (PoseLandmark.LEFT_HEEL,  PoseLandmark.LEFT_FOOT_INDEX, (0, 255, 0)),

    # Right leg -> yellow (BGR)
    (PoseLandmark.RIGHT_HIP,   PoseLandmark.RIGHT_KNEE,       (0, 255, 255)),
    (PoseLandmark.RIGHT_KNEE,  PoseLandmark.RIGHT_ANKLE,      (0, 255, 255)),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL,       (0, 255, 255)),
    (PoseLandmark.RIGHT_HEEL,  PoseLandmark.RIGHT_FOOT_INDEX, (0, 255, 255)),
]


# --------- Core helpers ---------
def draw_body_only(
    frame,
    landmarks: Sequence,
    img_w: int,
    img_h: int,
    visibility_thr: float = 0.5,
    point_radius: int = 3,
    point_color: ColorBGR = (0, 255, 0),  # green dots
    point_thickness: int = -1,
    line_thickness: int = 2,
) -> None:
    """
    Draw body-only landmarks and connections using OpenCV.
    - `landmarks` is the list from MediaPipe (result.pose_landmarks.landmark)
    - Silent no-op if visibility is low or points are off-frame.
    """
    # Points
    for idx in BODY_IDX:
        lm = landmarks[idx]
        if getattr(lm, "visibility", 1.0) < visibility_thr:
            continue
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        if 0 <= x < img_w and 0 <= y < img_h:
            cv2.circle(frame, (x, y), point_radius, point_color, point_thickness)

    # Lines
    for a, b, color in BODY_CONNECTIONS:
        la, lb = landmarks[a.value], landmarks[b.value]
        if getattr(la, "visibility", 1.0) < visibility_thr:
            continue
        if getattr(lb, "visibility", 1.0) < visibility_thr:
            continue
        ax, ay = int(la.x * img_w), int(la.y * img_h)
        bx, by = int(lb.x * img_w), int(lb.y * img_h)
        if (0 <= ax < img_w and 0 <= ay < img_h and
            0 <= bx < img_w and 0 <= by < img_h):
            cv2.line(frame, (ax, ay), (bx, by), color, line_thickness)


def dump_body_to_csv(
    landmarks: Sequence,
    frame_idx: int,
    csv_writer: csv.writer
) -> None:
    """
    Write only body landmarks to CSV:
    columns: Frame, LandmarkIndex, BodyPartName, x, y, z, Visibility
    """
    for idx in BODY_IDX:
        lm = landmarks[idx]
        lm_name = PoseLandmark(idx).name
        visibility = getattr(lm, "visibility", 1.0)
        csv_writer.writerow([frame_idx, idx, lm_name, lm.x, lm.y, lm.z, visibility])


def open_video_writer(
    out_path: Path,
    fps: float,
    size: Tuple[int, int],
    fourcc: str = 'mp4v'
) -> cv2.VideoWriter:
    """
    Create a cv2.VideoWriter and validate it's open. Raises RuntimeError if it fails.
    """
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(out_path), fourcc_code, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")
    return writer

MP_INDICES = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
}

GROUP_INDICES = {
    'left_arm':     [MP_INDICES['left_shoulder'],   MP_INDICES['left_elbow'],   MP_INDICES['left_wrist']],
    'right_arm':    [MP_INDICES['right_shoulder'],  MP_INDICES['right_elbow'],  MP_INDICES['right_wrist']],
    'left_leg':     [MP_INDICES['left_hip'],        MP_INDICES['left_knee'],    MP_INDICES['left_ankle']],
    'right_leg':    [MP_INDICES['right_hip'],       MP_INDICES['right_knee'],   MP_INDICES['right_ankle']],
    'torso':        [MP_INDICES['left_hip'],        MP_INDICES['right_hip'],    MP_INDICES['left_shoulder'], MP_INDICES['right_shoulder']]
}

DIMS = {
    'left_arm': 6,
    'right_arm': 6,
    'left_leg': 6,
    'right_leg': 6,
    'torso': 8
}

def extract_body_parts_from_mp(landmarks: Sequence) -> Tuple[np.array, ...]:

    try:
        kpt_la = np.array([[landmarks[i].x, landmarks[i].y] for i in GROUP_INDICES['left_arm']]).flatten()
        kpt_ra = np.array([[landmarks[i].x, landmarks[i].y] for i in GROUP_INDICES['right_arm']]).flatten()
        kpt_t  = np.array([[landmarks[i].x, landmarks[i].y] for i in GROUP_INDICES['torso']]).flatten()
        kpt_ll = np.array([[landmarks[i].x, landmarks[i].y] for i in GROUP_INDICES['left_leg']]).flatten()
        kpt_rl = np.array([[landmarks[i].x, landmarks[i].y] for i in GROUP_INDICES['right_leg']]).flatten()

        return kpt_la, kpt_ra, kpt_t, kpt_ll, kpt_rl

    except Exception:
        return (np.zeros(DIMS['left_arm']), 
                np.zeros(DIMS['right_arm']), 
                np.zeros(DIMS['torso']), 
                np.zeros(DIMS['left_leg']), 
                np.zeros(DIMS['right_leg']))
    
def draw_predictions(frame: np.ndarray, probs:np.ndarray, class_names: List[str]) -> None:
    pred_idx = np.argmax(probs)
    pred_name = class_names[pred_idx]
    confidence = probs[pred_idx]

    # Draw prediction on frame
    text = f"{pred_name.upper()} {confidence:.2f}"
    color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)
    if pred_name == "neutral":
        color = (255, 255, 255)

    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Drawing probabilities bars
    for i, prob in enumerate(probs):
        bar_x = 50
        bar_y = 100 + i * 40
        bar_width = int(prob*100)

        bar_color = (200, 0 ,0)
        if i == pred_idx:
            bar_color = color
        elif class_names[i] == 'neutral':
            bar_color = (150, 150, 150)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 25), bar_color, -1)
        cv2.putText(frame, f"{class_names[i]:{prob:.2f}}", (bar_x + 5, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def run_dataset_generation(in_path: Path, args: ArgumentParser):

    print(f"Mode: Dataset generation. Output in: {args.outdir}")
    
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    annotated_path: Path = outdir / f"{in_path.stem}_Landmarked.mp4"
    csv_path: Path = outdir / f"{in_path.stem}_Landmarks.csv"

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = open_video_writer(annotated_path, fps, (width, height), fourcc='mp4v')

    write_csv = not args.no_csv
    csv_file = None
    csv_writer = None
    if write_csv:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Frame", "Landmark", "Body Part", "x Pos", "y Pos", "z Pos", "Visibility"])

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=False, model_complexity=args.model_complexity) as pose:
        frame_idx = 0
        with tqdm(total=total_frames, desc="Dataset generation", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    lms = result.pose_landmarks.landmark
                    draw_body_only(
                        frame, lms, width, height,
                        visibility_thr=args.visibility_thr
                    )
                    if write_csv and csv_writer is not None:
                        dump_body_to_csv(lms, frame_idx, csv_writer)

                writer.write(frame)
                frame_idx += 1
                pbar.update(1)

    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()

    print(f"\nVideo saved in: {annotated_path}")
    if write_csv:
        print(f"CSV saved in: {csv_path}")

def run_inference(in_path: Path, args: ArgumentParser):

    print("Mode: Real-time inference.")

    # Load Keras model
    if not args.model_path:
        raise ValueError("For inference, --model_path is required (e.g. --model_path Shot_Classification.keras)")
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    # Disable TensorFlow logging except for errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    model = tf.keras.models.load_model(model_path)
    class_names = ['backhand', 'forehand', 'neutral']
    T = model.input_shape[0][1]
    print(f"Model loaded. Sequence length: {T} frame.")

    # Load standardization values
    std_path = Path('Standardization_Values.npz')
    if not std_path.exists():
        raise FileNotFoundError(f"File 'Standardization_Values.npz' not found. "
                                f"Please run the notebook first to save it.")

    std_data = np.load(std_path)
    std_params = {key: std_data[key] for key in std_data}
    print("Standardization values loaded.")

    # Initialize buffers (deque)
    buffers = {
        'la': deque([np.zeros(DIMS['left_arm'])] * T, maxlen=T),
        'ra': deque([np.zeros(DIMS['right_arm'])] * T, maxlen=T),
        't':  deque([np.zeros(DIMS['torso'])] * T, maxlen=T),
        'll': deque([np.zeros(DIMS['left_leg'])] * T, maxlen=T),
        'rl': deque([np.zeros(DIMS['right_leg'])] * T, maxlen=T),
    }

    # Open video and MediaPipe
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {in_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    mp_pose = mp.solutions.pose
    
    with mp_pose.Pose(static_image_mode=False, model_complexity=args.model_complexity) as pose:
        with tqdm(total=total_frames, desc="Inference", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                # Initialize features to zero
                features = {
                    'la': np.zeros(DIMS['left_arm']), 'ra': np.zeros(DIMS['right_arm']),
                    't': np.zeros(DIMS['torso']), 'll': np.zeros(DIMS['left_leg']),
                    'rl': np.zeros(DIMS['right_leg'])
                }

                if result.pose_landmarks:
                    lms = result.pose_landmarks.landmark

                    # Extract features
                    features_la, features_ra, features_t, features_ll, features_rl = \
                        extract_body_parts_from_mp(lms)

                    # Draw skeleton
                    draw_body_only(
                        frame, lms, width, height, 
                        visibility_thr=args.visibility_thr
                    )

                    # Standardize features *before* adding to buffer
                    features = {
                        'la': (features_la - std_params['mu_la']) / std_params['sd_la'],
                        'ra': (features_ra - std_params['mu_ra']) / std_params['sd_ra'],
                        't':  (features_t  - std_params['mu_t'])  / std_params['sd_t'],
                        'll': (features_ll - std_params['mu_ll']) / std_params['sd_ll'],
                        'rl': (features_rl - std_params['mu_rl']) / std_params['sd_rl'],
                    }

                # Update buffers (with zeros or standardized features)
                buffers['la'].append(features['la'])
                buffers['ra'].append(features['ra'])
                buffers['t'].append(features['t'])
                buffers['ll'].append(features['ll'])
                buffers['rl'].append(features['rl'])

                # Prepare model input (add batch dimension)
                model_input = [
                    np.array([buffers['la']]),
                    np.array([buffers['ra']]),
                    np.array([buffers['t']]),
                    np.array([buffers['ll']]),
                    np.array([buffers['rl']])
                ]

                # Predict RNN
                probs = model.predict(model_input, verbose=0)[0]

                # Draw results
                draw_predictions(frame, probs, class_names)
                
                cv2.imshow("Inferenza Shot Classification RNN", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()

# --------- Main pipeline ---------
def main():
    parser = ArgumentParser(description="Analyze tennis video with MediaPipe. "
                                        "Save CSV (default) or run RNN inference (with --infer).")
    # Common arguments
    parser.add_argument("video", type=str, help="Input video path")
    parser.add_argument(
        "--visibility-thr", type=float, default=0.5,
        help="Visibility threshold for landmark (default: 0.5)"
    )
    parser.add_argument(
        "--model-complexity", type=int, default=1, choices=[0, 1, 2],
        help="MediaPipe Pose model complexity (0,1,2). Default: 1"
    )

    # Arguments for "Dataset Generation" mode
    parser.add_argument(
        "--outdir", type=Path, default=Path("Landmarked"),
        help="Output directory for video and CSV (default: ./Landmarked)"
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Disable CSV export (only annotated video)."
    )

    # Arguments for "Inference" mode
    parser.add_argument(
        "--infer", action="store_true",
        help="Activate real-time inference mode (disables CSV and video output)."
    )
    parser.add_argument(
        "--model_path", type=str, default="Shot_Classification.keras",
        help="Path to the .keras model for inference (default: Shot_Classification.keras)"
    )
    
    args = parser.parse_args()
    
    in_path = Path(args.video)
    if not in_path.exists():
        raise FileNotFoundError(f"Input video not found: {in_path}")

    # Choose which function to execute
    if args.infer:
        run_inference(in_path, args)
    else:
        run_dataset_generation(in_path, args)

if __name__ == "__main__":
    main()