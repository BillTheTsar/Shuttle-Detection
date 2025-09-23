# video_inferencer.py
import argparse
import time
from pathlib import Path
from collections import deque
import numpy as np

import cv2
import csv
import torch

from engine_trt import Engine  # your engine returns (gx, gy, gv, calib_triggered)
from tensorRT_wrapper import TRTWrapperTorch
# from model import Stage1ModelBiFPN, Stage2ModelHeatmap, Stage2ModelBiFPN

def main(video_path: str, out_csv: str, device="cuda", mode="smart"):
    # ----------------- load TRT engines -----------------
    print("Loading TensorRT engines...")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    stage1 = TRTWrapperTorch(
        str(PROJECT_ROOT / "trt_engines/stage1_v4_fp16.engine"),
        input_names=["current", "past", "positions"],
        output_names=["xy", "heatmap"],
        device=device
    )

    stage2_main = TRTWrapperTorch(
        str(PROJECT_ROOT / "trt_engines/stage2_v4_fp16.engine"),
        input_names=["current_crop", "past_crops", "positions"],
        output_names=["xy", "vis_logit", "heatmap"],
        device=device
    )

    stage2_supp = TRTWrapperTorch(
        str(PROJECT_ROOT / "trt_engines/stage2_v7_fp16.engine"),
        input_names=["current_crop", "past_crops", "positions"],
        output_names=["xy", "heatmap"],
        device=device
    )
    print("✅ Engines loaded")

    # ----------------- build Engine -----------------
    engine = Engine(
        stage1, stage2_main, stage2_supp,
        frame_hw=(1080, 1920),
        triangle_radius=0.06,
        agree_thresh=0.05,
        jump_thresh=0.03,
        angle_thresh=3.14159 / 2,
        vis_thresh=0.65,
        invisible_trigger_len=5,
        fast_calib_interval=20,
        mode=mode,
        device=device,
        use_amp=False,  # AMP is handled internally by TRT now
    )

    # Optional warmup (you said you commented warmup in engine; leaving this off)
    # engine.warmup(8)

    # ----------------- video loop -----------------
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    # last 30 outputs used by the engine
    positions_deque = deque([[0.0, 0.0, 0]] * 30, maxlen=30)

    # sliding window of size 6 storing whether a *smart-mode* calibration was triggered
    calib_window = deque(maxlen=6)

    # buffer last ≤6 rows before writing to CSV so we can retroactively overwrite
    # each row is (frame_idx, x, y, vis)
    out_window = deque()

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    csvfile = open(out_csv, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["frame", "x", "y", "vis"])  # header

    # Debug counters
    rule1_count = 0
    rule2_count = 0

    frame_idx = 0
    t0 = time.time()

    def flush_left_if_needed():
        """Write the oldest row if our out_window exceeds 6 rows."""
        if len(out_window) > 6:
            row = out_window.popleft()
            writer.writerow(row)

    def flush_all():
        """Write everything remaining at the end."""
        while out_window:
            writer.writerow(out_window.popleft())

    while True:
        # t_read0 = time.perf_counter()
        ret, frame_bgr = cap.read()
        # t_read1 = time.perf_counter()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        # t_rgb = time.perf_counter()

        # Engine step: always returns 4 values; in non-smart modes calib_triggered is False (by design)
        gx, gy, gv, calib_triggered = engine.step(frame_rgb, positions_deque)
        # torch.cuda.synchronize()
        # t_step = time.perf_counter()
        # print(f"read: {(t_read1-t_read0)*1000:.2f} ms. rgb: {(t_rgb-t_read1)*1000:.2f} ms. step: {(t_step-t_rgb)*1000:.2f} ms")
        # Stage result for output (we buffer before writing to CSV)
        out_window.append((frame_idx, gx, gy, gv))
        # After any rewrites, we can safely flush the oldest row if window > 6
        flush_left_if_needed()
        # Append to the engine's positions history (most recent at end)
        positions_deque.append([gx, gy, gv])

        # ----------------- RULE 2: sliding window overwrite (smart mode only) -----------------
        # Count calibrations over the last 6 frames that are *explicit smart-mode* calibrations
        if mode == "smart":
            calib_window.append(bool(calib_triggered))
            if sum(calib_window) >= 4:
                # Overwrite window of 6 frames: previous 5 + current
                n = len(calib_window)
                for i in range(n): # If calib_window = [True, True, True], we only remove 3, not 6
                    # 1) positions_deque: last up to 6 entries → (0,0,0)
                    positions_deque[-(i + 1)] = [0.0, 0.0, 0]
                    # 2) out_window: set all buffered rows to zero
                    f, _, _, _ = out_window[-(i + 1)]
                    out_window[-(i + 1)] = (f, 0.0, 0.0, 0)
                # 3) current frame's verdict → zero
                gx, gy, gv = 0.0, 0.0, 0
                # After applying, clear window to avoid immediate re-firing
                calib_window.clear()
                rule2_count += 1

        # ----------------- RULE 1: short visible-block overwrite -----------------
        # If the current frame is invisible, inspect the *immediately preceding* visible block.
        if gv == 0:
            # Walk backwards from the entry just before current to find a contiguous visible block
            block_len = 0
            for p in reversed(list(positions_deque)[:-1]):
                if int(p[2]) == 1:
                    block_len += 1
                    if block_len >= 5:   # stop early; we only overwrite blocks < 5
                        break
                else:
                    break
            if 0 < block_len < 5:
                # Overwrite that block *and* the current invisible frame → total block_len + 1 rows
                total = block_len + 1

                # 1) positions_deque: last 'total' entries → (0,0,0)
                for i in range(total):
                    positions_deque[-(i + 1)] = [0.0, 0.0, 0]

                # 2) out_window: rewrite last 'total' rows to zeros
                m = min(total, len(out_window))
                for i in range(m):
                    f, _, _, _ = out_window[-(i + 1)]
                    out_window[-(i + 1)] = (f, 0.0, 0.0, 0)
                rule1_count += 1

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")
            print(f"- Big angles: {engine.big_angle}")
            print(f"- Big jumps: {engine.big_jump}")
            print(f"- Freshly invisible: {engine.freshly_invisible}")
            print(f" - Rule 1 short-block wipes: {rule1_count}")
            print(f" - Rule 2 window resets: {rule2_count}")
            engine.big_angle = 0
            engine.big_jump = 0
            engine.freshly_invisible = 0
            rule1_count = 0
            rule2_count = 0

    # flush remaining rows
    flush_all()

    dt = time.time() - t0
    fps = frame_idx / dt if dt > 0 else 0.0
    print(f"✅ Done {frame_idx} frames in {dt:.1f}s ({fps:.2f} FPS)")

    cap.release()
    csvfile.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Label every frame of a video with shuttle head predictions."
    )

    ap.add_argument(
        "--video",
        type=str,
        default="E:/AI/Badminton Landing Project/Playing footage all angles/5.mp4",
        help="Path to the input video file to be processed."
    )

    ap.add_argument(
        "--csv",
        type=str,
        default="D:/Shuttle Detection Model/outputs/video_inference_csv/5_csv_smart.csv",
        help="Path to the output CSV file where predictions will be saved."
    )

    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g. 'cuda' for GPU or 'cpu' for CPU)."
    )

    ap.add_argument(
        "--mode",
        type=str,
        choices=["calib", "fast", "smart"],
        default="smart",
        help=(
            "Inference mode:\n"
            "  'calib' - calibration mode, usually slower but useful for debugging;\n"
            "  'fast'  - prioritize speed over accuracy;\n"
            "  'smart' - balanced mode (recommended)."
        )
    )

    args = ap.parse_args()

    main(args.video, args.csv, device=args.device, mode=args.mode)