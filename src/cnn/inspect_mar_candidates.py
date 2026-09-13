from pathlib import Path

import cv2
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

YAWDD_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "YawDD.rar"
    / "YawDD"
    / "YawDD dataset"
)

CSV_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "yawdd_mar"
    / "yawdd_mar_samples.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "yawdd_mar"
    / "candidate_frames"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CANDIDATE MAR LEVELS
# ============================================================

TARGETS = {
    "Normal": [0.10, 0.20, 0.30],
    "Talking": [0.30, 0.40, 0.50, 0.60, 0.70],
    "Yawning": [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20],
}


# ============================================================
# FIND VIDEO
# ============================================================

def find_video(video_name):

    matches = list(
        YAWDD_ROOT.rglob(video_name)
    )

    if not matches:
        return None

    return matches[0]


# ============================================================
# GET FRAME
# ============================================================

def read_frame(video_path, frame_number):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        return None

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_number)
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        return None

    return frame


# ============================================================
# FIND CLOSEST FRAMES
# ============================================================

def find_closest(df, behavior, target):

    subset = df[
        df["behavior"] == behavior
    ].copy()

    if subset.empty:
        return None

    subset["distance"] = (
        subset["mar"] - target
    ).abs()

    row = subset.sort_values(
        "distance"
    ).iloc[0]

    return row


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MAR CANDIDATE FRAME INSPECTION")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)

    print(
        f"Loaded {len(df)} MAR samples"
    )

    saved = 0

    for behavior, targets in TARGETS.items():

        behavior_dir = (
            OUTPUT_DIR
            / behavior.lower()
        )

        behavior_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for target in targets:

            row = find_closest(
                df,
                behavior,
                target
            )

            if row is None:
                continue

            video_name = row["video"]
            frame_number = int(
                row["frame"]
            )
            actual_mar = float(
                row["mar"]
            )

            video_path = find_video(
                video_name
            )

            if video_path is None:
                print(
                    f"Video not found: "
                    f"{video_name}"
                )
                continue

            frame = read_frame(
                video_path,
                frame_number
            )

            if frame is None:
                print(
                    f"Frame read failed: "
                    f"{video_name} "
                    f"frame {frame_number}"
                )
                continue

            # Add information to image
            display = frame.copy()

            text = (
                f"{behavior} | "
                f"MAR={actual_mar:.3f} | "
                f"frame={frame_number}"
            )

            cv2.putText(
                display,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            output_name = (
                f"{behavior.lower()}_"
                f"target_{target:.2f}_"
                f"actual_{actual_mar:.3f}_"
                f"frame_{frame_number}.jpg"
            )

            output_path = (
                behavior_dir
                / output_name
            )

            cv2.imwrite(
                str(output_path),
                display
            )

            saved += 1

            print(
                f"{behavior:10s} | "
                f"target={target:.2f} | "
                f"actual={actual_mar:.3f} | "
                f"frame={frame_number}"
            )

    print("\n" + "=" * 70)
    print(f"Saved {saved} candidate frames")
    print("=" * 70)

    print(
        f"\nOpen this folder:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()