from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import cv2

##########################################################################################################
# Script to annotate a video with tennis shots (Forehand, Backhand, Serve),
# and export a CSV file with the annotations.
# Usage:
#   python Labeler.py input_video.mp4
# Outputs:
#   - CSV file with annotations in ./Shots_Annotation/input_video_Labeled.csv
##########################################################################################################


LEFT_KEYS  = {81, 2424832}  # frame back
UP_KEYS    = {82, 2490368}  # forehand
RIGHT_KEYS = {83, 2555904}  # frame forward
DOWN_KEYS  = {84, 2621440}  # backhand
SPACEBAR   = 32             # serve
ESCAPE     = 27             # exit

def draw_legend(frame, user_input, total_frames):
    legend = [
        "LEGEND:",
        "LEFT ARROW : Go back 5 frames",
        "RIGHT ARROW : Go forward 5 frames",
        "UP ARROW : Annotate Forehand",
        "DOWN ARROW : Annotate Backhand",
        "SPACE : Annotate Serve",
        "ESC : Exit and save annotations"
    ]

    x, y0, dy = 10, 20, 20

    for i, line in enumerate(legend):
        y = y0 + i * dy
        cv2.putText(frame, line, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA) # black border
        cv2.putText(frame, line, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA) # white legend text
    
    y_frame = y0 + len(legend) * dy
    cv2.putText(frame, f"CURRENT FRAME: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{total_frames}", (x,y_frame), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA) # white frame counter  

    if user_input:
        y_log = y0 + len(legend) * dy + 20
        cv2.putText(frame, user_input, (x,y_log), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA) # white user input log
    
    return frame

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Annotate a video and write a csv file containing tennis shots"
    )
    parser.add_argument("video")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("Error opening video file")

    df = pd.DataFrame(columns=["Frame", "Shot"])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    shots_list = []

    user_log = ""
    
    while cap.isOpened():
        frame_id = max(0, min(frame_id, max(0, total_frames - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = cap.read()

        if not ret:
            print("End of video reached")
            break 
        
        k = cv2.waitKeyEx(30)

        if k in LEFT_KEYS:
            frame_id -= 5
            print("Going back 5 frames")
            continue
        elif k in RIGHT_KEYS:
            frame_id += 5
            print("Going forward 5 frames")
            continue
        elif k in UP_KEYS:
            shots_list.append({"Shot": "Forehand", "Frame": frame_id})
            df = pd.DataFrame.from_records(shots_list)
            user_log = "Added Forehand at frame {}".format(frame_id)
            print("Added Forehand at frame", frame_id)
        elif k in DOWN_KEYS:
            shots_list.append({"Shot": "Backhand", "Frame": frame_id})
            df = pd.DataFrame.from_records(shots_list)
            user_log = "Added Backhand at frame {}".format(frame_id)
            print("Added Backhand at frame", frame_id)
        elif k == SPACEBAR:
            shots_list.append({"Shot": "Serve", "Frame": frame_id})
            df = pd.DataFrame.from_records(shots_list)
            user_log = "Added Serve at frame {}".format(frame_id)
            print("Added Serve at frame", frame_id)
        elif k == ESCAPE:
            break

        frame_id += 1

        frame = draw_legend(frame, f"USER INPUT LOG: {user_log}", total_frames)

        cv2.imshow("Frame", frame)

    out_dir = Path("Shots_Annotation")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{Path(args.video).stem}_Labeled.csv"
    
    df.to_csv(out_file, index=False)
    print(f"Annotation file was written to {out_file}")
