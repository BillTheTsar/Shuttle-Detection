import cv2
import pandas as pd
from pathlib import Path
import argparse

def main(video_path, csv_path, output_path):
    # --- Load predictions ---
    df = pd.read_csv(csv_path)

    # --- Open video ---
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

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


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    ap = argparse.ArgumentParser(
        description="Overlay predicted shuttle positions onto a video."
    )
    ap.add_argument(
        "--video",
        type=str,
        default="E:/AI/Badminton Landing Project/Playing footage all angles/3.mp4",
        help="Path to input video file.",
    )
    ap.add_argument(
        "--csv",
        type=str,
        default=PROJECT_ROOT / "outputs/video_inference_csv/3.csv",
        help="Path to CSV file with predictions (columns: frame, x, y, vis).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=PROJECT_ROOT / "outputs/video_inference_video/3.mp4",
        help="Path to save annotated video.",
    )
    args = ap.parse_args()

    main(args.video, args.csv, args.out)