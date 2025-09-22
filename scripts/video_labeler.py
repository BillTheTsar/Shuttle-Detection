import cv2
import pandas as pd
from pathlib import Path

# --- Inputs ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
csv_path = PROJECT_ROOT / "outputs/video_inference_csv/3.csv"
video_path = "E:/AI/Badminton Landing Project/Playing footage all angles/3.mp4"
output_path = PROJECT_ROOT / "outputs/video_inference_video/3.mp4"

# --- Load predictions ---
df = pd.read_csv(csv_path)

# --- Open video ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Get prediction row for this frame
    row = df[df["frame"] == frame_idx]
    if not row.empty and row.iloc[0]["vis"] == 1:
        # Convert normalized coords -> pixel coords
        x_px = int(row.iloc[0]["x"] * W)
        y_px = int(row.iloc[0]["y"] * H)

        # Draw a circle on the prediction
        cv2.circle(frame, (x_px, y_px), 4, (0, 0, 255), -1)  # red dot

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"Saved annotated video to {output_path}")