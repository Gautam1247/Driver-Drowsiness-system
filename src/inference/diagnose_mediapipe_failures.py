from pathlib import Path
import random

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACE_MODEL = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)

TEST_IMAGE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "yolo_dms"
    / "images"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "mediapipe_diagnosis"
)

DETECTED_DIR = OUTPUT_DIR / "detected"
FAILED_DIR = OUTPUT_DIR / "failed"

DETECTED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FAILED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_IMAGES = 20
RANDOM_SEED = 42

# Same MediaPipe settings used in our latest batch test.
MIN_FACE_DETECTION_CONFIDENCE = 0.30
MIN_FACE_PRESENCE_CONFIDENCE = 0.30


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MEDIAPIPE FACE DETECTION FAILURE DIAGNOSIS")
print("=" * 70)

print(f"Project root : {PROJECT_ROOT}")
print(f"Test images  : {TEST_IMAGE_DIR}")
print(f"Output       : {OUTPUT_DIR}")
print(f"Images       : {NUM_IMAGES}")
print(f"Random seed  : {RANDOM_SEED}")

print(
    f"Detection confidence : "
    f"{MIN_FACE_DETECTION_CONFIDENCE}"
)

print(
    f"Presence confidence  : "
    f"{MIN_FACE_PRESENCE_CONFIDENCE}"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not FACE_MODEL.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found:\n{FACE_MODEL}"
    )

if not TEST_IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Test image directory not found:\n{TEST_IMAGE_DIR}"
    )


# ============================================================
# FIND TEST IMAGES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

all_images = sorted(
    p
    for p in TEST_IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
)

print(
    f"\nTotal test images available: "
    f"{len(all_images)}"
)

if len(all_images) < NUM_IMAGES:
    raise RuntimeError(
        f"Only {len(all_images)} images available, "
        f"but {NUM_IMAGES} requested."
    )


# ============================================================
# SELECT SAME 20 IMAGES AS BATCH TEST
# ============================================================

random.seed(RANDOM_SEED)

selected_images = random.sample(
    all_images,
    NUM_IMAGES
)

print(
    f"Selected {len(selected_images)} images "
    f"using seed {RANDOM_SEED}."
)


# ============================================================
# LOAD MEDIAPIPE
# ============================================================

print("\n" + "=" * 70)
print("LOADING MEDIAPIPE FACE LANDMARKER")
print("=" * 70)

base_options = python.BaseOptions(
    model_asset_path=str(FACE_MODEL)
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=(
        MIN_FACE_DETECTION_CONFIDENCE
    ),
    min_face_presence_confidence=(
        MIN_FACE_PRESENCE_CONFIDENCE
    ),
)

face_landmarker = (
    vision.FaceLandmarker.create_from_options(
        options
    )
)

print("MediaPipe loaded successfully.")


# ============================================================
# PROCESS IMAGES
# ============================================================

detected_count = 0
failed_count = 0
read_failure_count = 0
inference_failure_count = 0

failed_images = []
detected_images = []


print("\n" + "=" * 70)
print("RUNNING MEDIAPIPE DIAGNOSIS")
print("=" * 70)


for index, image_path in enumerate(
    selected_images,
    start=1
):

    print(
        f"\n[{index:02d}/{NUM_IMAGES}] "
        f"{image_path.name}"
    )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print("  Status: IMAGE READ FAILURE")

        read_failure_count += 1

        continue


    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    try:

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        detection_result = (
            face_landmarker.detect(
                mp_image
            )
        )

    except Exception as e:

        print(
            "  Status: MEDIAPIPE ERROR"
        )

        print(
            f"  Error: {e}"
        )

        inference_failure_count += 1

        failed_images.append(
            image_path.name
        )

        # Save original image
        output_path = (
            FAILED_DIR
            / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        continue


    # --------------------------------------------------------
    # Check face detection
    # --------------------------------------------------------

    if not detection_result.face_landmarks:

        print(
            "  Status: FACE NOT DETECTED"
        )

        failed_count += 1

        failed_images.append(
            image_path.name
        )

        # Save original image unchanged.
        output_path = (
            FAILED_DIR
            / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        continue


    # --------------------------------------------------------
    # Face detected
    # --------------------------------------------------------

    landmarks = (
        detection_result.face_landmarks[0]
    )

    detected_count += 1

    detected_images.append(
        image_path.name
    )

    print(
        "  Status: FACE DETECTED"
    )

    print(
        f"  Landmarks: {len(landmarks)}"
    )


    # ========================================================
    # CREATE ANNOTATED IMAGE
    # ========================================================

    annotated = image.copy()

    height, width = image.shape[:2]


    # --------------------------------------------------------
    # Draw face landmarks
    # --------------------------------------------------------

    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        # Ignore landmarks outside image.
        if (
            x < 0
            or x >= width
            or y < 0
            or y >= height
        ):
            continue

        cv2.circle(
            annotated,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )


    # --------------------------------------------------------
    # Estimate face bounding box
    # --------------------------------------------------------

    face_points = []

    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        if (
            0 <= x < width
            and 0 <= y < height
        ):
            face_points.append(
                (x, y)
            )


    if face_points:

        xs = [
            point[0]
            for point in face_points
        ]

        ys = [
            point[1]
            for point in face_points
        ]

        face_x1 = min(xs)
        face_y1 = min(ys)
        face_x2 = max(xs)
        face_y2 = max(ys)

        cv2.rectangle(
            annotated,
            (face_x1, face_y1),
            (face_x2, face_y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            annotated,
            "FACE DETECTED",
            (
                face_x1,
                max(face_y1 - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # Save annotated image
    # --------------------------------------------------------

    output_path = (
        DETECTED_DIR
        / image_path.name
    )

    cv2.imwrite(
        str(output_path),
        annotated
    )


# ============================================================
# CLEANUP
# ============================================================

face_landmarker.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)

print(
    f"Images selected       : {NUM_IMAGES}"
)

print(
    f"Faces detected        : {detected_count}"
)

print(
    f"Face detection failed : {failed_count}"
)

print(
    f"Image read failures   : {read_failure_count}"
)

print(
    f"MediaPipe errors      : {inference_failure_count}"
)


# ============================================================
# FAILED IMAGES
# ============================================================

print("\n" + "=" * 70)
print("FAILED IMAGES")
print("=" * 70)

if failed_images:

    for index, filename in enumerate(
        failed_images,
        start=1
    ):

        print(
            f"{index:02d}. {filename}"
        )

else:

    print("No MediaPipe failures.")


# ============================================================
# DETECTED IMAGES
# ============================================================

print("\n" + "=" * 70)
print("DETECTED IMAGES")
print("=" * 70)

if detected_images:

    for index, filename in enumerate(
        detected_images,
        start=1
    ):

        print(
            f"{index:02d}. {filename}"
        )

else:

    print("No faces detected.")


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    f"Detected images:\n{DETECTED_DIR}"
)

print(
    f"\nFailed images:\n{FAILED_DIR}"
)

print("\n" + "=" * 70)
print("MEDIAPIPE DIAGNOSIS COMPLETED")
print("=" * 70)