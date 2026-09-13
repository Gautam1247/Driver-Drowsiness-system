from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)

YAWDD_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "YawDD.rar"
    / "YawDD"
    / "YawDD dataset"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "yawdd_mar"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = FaceLandmarker.create_from_options(
    options
)


# ============================================================
# MOUTH LANDMARKS
# ============================================================

LEFT_MOUTH = 61
RIGHT_MOUTH = 291
UPPER_MOUTH = 13
LOWER_MOUTH = 14


# ============================================================
# MAR
# ============================================================

def distance(p1, p2):
    return np.sqrt(
        (p1.x - p2.x) ** 2
        +
        (p1.y - p2.y) ** 2
    )


def calculate_mar(landmarks):

    horizontal = distance(
        landmarks[LEFT_MOUTH],
        landmarks[RIGHT_MOUTH]
    )

    vertical = distance(
        landmarks[UPPER_MOUTH],
        landmarks[LOWER_MOUTH]
    )

    if horizontal < 1e-6:
        return None

    return vertical / horizontal


# ============================================================
# DETECT MAR FOR FRAME
# ============================================================

def get_mar(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]

    return calculate_mar(landmarks)


# ============================================================
# FIND MIRROR VIDEOS
# ============================================================

def get_videos():

    videos = []

    for path in YAWDD_ROOT.rglob("*.avi"):

        name = path.name.lower()

        if "normal" in name:
            behavior = "Normal"

        elif "talking" in name:
            behavior = "Talking"

        elif "yawning" in name:
            behavior = "Yawning"

        else:
            continue

        videos.append(
            {
                "path": path,
                "behavior": behavior
            }
        )

    return sorted(
        videos,
        key=lambda x: str(x["path"])
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    videos = get_videos()

    print("=" * 70)
    print("YawDD MAR DISTRIBUTION ANALYSIS")
    print("=" * 70)

    print(f"Videos found: {len(videos)}")

    behavior_counts = defaultdict(int)

    for video in videos:
        behavior_counts[
            video["behavior"]
        ] += 1

    print("\nVideo distribution:")

    for behavior in [
        "Normal",
        "Talking",
        "Yawning"
    ]:

        print(
            f"{behavior:10s}: "
            f"{behavior_counts[behavior]}"
        )


    # --------------------------------------------------------
    # Analyze frames
    # --------------------------------------------------------

    rows = []

    for video_idx, video_info in enumerate(
        tqdm(
            videos,
            desc="Processing videos"
        )
    ):

        video_path = video_info["path"]
        behavior = video_info["behavior"]

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            print(
                f"\nWARNING: Could not open "
                f"{video_path}"
            )
            continue

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        # ----------------------------------------------------
        # Sample approximately 5 FPS
        # ----------------------------------------------------

        if fps <= 0:
            fps = 30.0

        step = max(
            1,
            int(round(fps / 5))
        )

        frame_index = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            if frame_index % step != 0:
                frame_index += 1
                continue

            mar = get_mar(frame)

            if mar is not None:

                rows.append(
                    {
                        "video": video_path.name,
                        "behavior": behavior,
                        "frame": frame_index,
                        "time_sec": (
                            frame_index / fps
                        ),
                        "mar": mar
                    }
                )

            frame_index += 1

        cap.release()


    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    csv_path = (
        OUTPUT_DIR
        / "yawdd_mar_samples.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("MAR ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Total sampled frames with face: "
        f"{len(df)}"
    )

    print(
        f"Saved CSV:\n{csv_path}"
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\nMAR statistics:")
    print("-" * 70)

    stats = (
        df
        .groupby("behavior")["mar"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            q90=lambda x: x.quantile(0.90),
            q95=lambda x: x.quantile(0.95),
            q99=lambda x: x.quantile(0.99),
            max="max"
        )
    )

    print(
        stats.to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    stats_path = (
        OUTPUT_DIR
        / "yawdd_mar_statistics.csv"
    )

    stats.to_csv(stats_path)

    print(
        f"\nStatistics saved:\n{stats_path}"
    )


    # --------------------------------------------------------
    # Threshold counts
    # --------------------------------------------------------

    print("\nFrame counts above MAR thresholds:")
    print("-" * 70)

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.60,
        0.70,
        0.80
    ]

    for threshold in thresholds:

        counts = (
            df.assign(
                above=df["mar"] >= threshold
            )
            .groupby("behavior")["above"]
            .agg(
                total="count",
                above="sum"
            )
        )

        print(
            f"\nMAR >= {threshold:.2f}"
        )

        for behavior in [
            "Normal",
            "Talking",
            "Yawning"
        ]:

            if behavior not in counts.index:
                continue

            total = counts.loc[
                behavior,
                "total"
            ]

            above = counts.loc[
                behavior,
                "above"
            ]

            percentage = (
                100.0 * above / total
            )

            print(
                f"  {behavior:10s}: "
                f"{int(above):6d} / "
                f"{int(total):6d} "
                f"({percentage:6.2f}%)"
            )


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    landmarker.close()


if __name__ == "__main__":
    main()