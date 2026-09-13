from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"

YAWDD_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "YawDD.rar"
    / "YawDD"
    / "YawDD dataset"
)

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "mouth_landmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


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

landmarker = FaceLandmarker.create_from_options(options)


# ============================================================
# MOUTH LANDMARK INDICES
# ============================================================

# Important landmarks around the mouth.
#
# 61  -> left mouth corner
# 291 -> right mouth corner
# 13  -> upper inner lip
# 14  -> lower inner lip

MOUTH_INDICES = [
    61,
    291,
    13,
    14,
]


# ============================================================
# MAR CALCULATION
# ============================================================

def euclidean_distance(p1, p2):
    return np.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def calculate_mar(landmarks):
    """
    Simple Mouth Aspect Ratio.

    MAR = vertical mouth opening / horizontal mouth width
    """

    left_corner = landmarks[61]
    right_corner = landmarks[291]

    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    horizontal = euclidean_distance(
        left_corner,
        right_corner
    )

    vertical = euclidean_distance(
        upper_lip,
        lower_lip
    )

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(frame):
    """
    Detect face landmarks and calculate MAR.
    """

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None, None

    landmarks = result.face_landmarks[0]

    mar = calculate_mar(landmarks)

    return landmarks, mar


# ============================================================
# FIND SAMPLE VIDEOS
# ============================================================

def find_sample_videos():
    videos = []

    for behavior in ["Normal", "Talking", "Yawning"]:

        matches = list(YAWDD_ROOT.rglob(f"*{behavior}*.avi"))

        if matches:
            videos.append((behavior, matches[0]))

    return videos


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MEDIAPIPE FACE LANDMARKER TEST")
    print("=" * 60)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Model        : {MODEL_PATH}")
    print(f"YawDD root   : {YAWDD_ROOT}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face Landmarker model not found:\n{MODEL_PATH}"
        )

    if not YAWDD_ROOT.exists():
        raise FileNotFoundError(
            f"YawDD directory not found:\n{YAWDD_ROOT}"
        )

    videos = find_sample_videos()

    print("\nSample videos found:")

    for behavior, path in videos:
        print(f"{behavior:10s} -> {path.name}")

    print("\nProcessing frames...\n")

    for behavior, video_path in videos:

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Could not open: {video_path}")
            continue

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        sample_positions = [
            int(total_frames * 0.25),
            int(total_frames * 0.50),
            int(total_frames * 0.75),
        ]

        print("-" * 60)
        print(f"Behavior : {behavior}")
        print(f"Video    : {video_path.name}")
        print(f"Frames   : {total_frames}")

        for i, frame_position in enumerate(sample_positions):

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_position
            )

            success, frame = cap.read()

            if not success:
                print(
                    f"  Frame {frame_position}: read failed"
                )
                continue

            landmarks, mar = process_frame(frame)

            if landmarks is None:
                print(
                    f"  Frame {frame_position}: "
                    f"NO FACE DETECTED"
                )
                continue

            print(
                f"  Frame {frame_position}: "
                f"face detected | "
                f"landmarks={len(landmarks)} | "
                f"MAR={mar:.4f}"
            )

            # Draw selected mouth landmarks
            output = frame.copy()

            h, w = output.shape[:2]

            for index in MOUTH_INDICES:

                point = landmarks[index]

                x = int(point.x * w)
                y = int(point.y * h)

                cv2.circle(
                    output,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Add MAR to image
            cv2.putText(
                output,
                f"MAR: {mar:.4f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            output_name = (
                f"{behavior.lower()}_"
                f"{i + 1}.jpg"
            )

            output_path = OUTPUT_DIR / output_name

            cv2.imwrite(
                str(output_path),
                output
            )

        cap.release()

    landmarker.close()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(f"Output directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()