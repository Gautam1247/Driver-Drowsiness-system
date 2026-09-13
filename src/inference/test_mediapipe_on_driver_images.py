from pathlib import Path
import random

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

YOLO_MODEL = (
    PROJECT_ROOT
    / "models"
    / "dms_yolov8n_best.pt"
)

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
    / "mediapipe_driver_test"
)

YOLO_DETECTED_DIR = (
    OUTPUT_DIR
    / "yolo_eye_detected"
)

MEDIAPIPE_DETECTED_DIR = (
    OUTPUT_DIR
    / "mediapipe_detected"
)

MEDIAPIPE_FAILED_DIR = (
    OUTPUT_DIR
    / "mediapipe_failed"
)


for directory in [
    YOLO_DETECTED_DIR,
    MEDIAPIPE_DETECTED_DIR,
    MEDIAPIPE_FAILED_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CONFIGURATION
# ============================================================

NUM_IMAGES = 20

RANDOM_SEED = 42

YOLO_CONFIDENCE = 0.25

MIN_FACE_DETECTION_CONFIDENCE = 0.30

MIN_FACE_PRESENCE_CONFIDENCE = 0.30


# YOLO classes
OPEN_EYE_CLASS = 0
CLOSED_EYE_CLASS = 1


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MEDIAPIPE TEST ON YOLO-SELECTED DRIVER IMAGES")
print("=" * 70)

print(
    f"Project root : {PROJECT_ROOT}"
)

print(
    f"YOLO model   : {YOLO_MODEL}"
)

print(
    f"Face model   : {FACE_MODEL}"
)

print(
    f"Test images  : {TEST_IMAGE_DIR}"
)

print(
    f"Images       : {NUM_IMAGES}"
)

print(
    f"Random seed  : {RANDOM_SEED}"
)

print(
    f"YOLO confidence : {YOLO_CONFIDENCE}"
)

print(
    f"MediaPipe detection confidence : "
    f"{MIN_FACE_DETECTION_CONFIDENCE}"
)

print(
    f"MediaPipe presence confidence  : "
    f"{MIN_FACE_PRESENCE_CONFIDENCE}"
)


# ============================================================
# CHECK FILES
# ============================================================

if not YOLO_MODEL.exists():
    raise FileNotFoundError(
        f"YOLO model not found:\n{YOLO_MODEL}"
    )

if not FACE_MODEL.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found:\n{FACE_MODEL}"
    )

if not TEST_IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Test image directory not found:\n"
        f"{TEST_IMAGE_DIR}"
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


# ============================================================
# LOAD YOLO
# ============================================================

print("\n" + "=" * 70)
print("1. LOADING YOLO")
print("=" * 70)

yolo_model = YOLO(
    str(YOLO_MODEL)
)

print("YOLO loaded successfully.")

print(
    f"Classes: {yolo_model.names}"
)


# ============================================================
# SELECT IMAGES WITH YOLO EYE DETECTIONS
# ============================================================

print("\n" + "=" * 70)
print("2. FINDING DRIVER-RELATED IMAGES")
print("=" * 70)

print(
    "Running YOLO over the DMS test set..."
)

eye_detected_images = []

yolo_failures = 0

for index, image_path in enumerate(
    all_images,
    start=1
):

    if index % 100 == 0:

        print(
            f"Processed {index}/{len(all_images)} images..."
        )

    try:

        results = yolo_model.predict(
            source=str(image_path),
            conf=YOLO_CONFIDENCE,
            device="cpu",
            verbose=False
        )

    except Exception as e:

        print(
            f"YOLO error on {image_path.name}: {e}"
        )

        yolo_failures += 1

        continue


    if not results:

        continue

    result = results[0]

    if result.boxes is None:
        continue

    detected_classes = (
        result.boxes.cls
        .cpu()
        .numpy()
        .astype(int)
        .tolist()
    )

    # Check whether YOLO detected either eye class.
    has_eye = any(
        cls in {
            OPEN_EYE_CLASS,
            CLOSED_EYE_CLASS
        }
        for cls in detected_classes
    )

    if has_eye:

        eye_detected_images.append(
            image_path
        )


print(
    "\nYOLO eye-detection filtering completed."
)

print(
    f"Images with YOLO eye detection : "
    f"{len(eye_detected_images)}"
)

print(
    f"YOLO processing failures        : "
    f"{yolo_failures}"
)


# ============================================================
# CHECK ENOUGH IMAGES
# ============================================================

if len(eye_detected_images) < NUM_IMAGES:

    raise RuntimeError(
        f"Only {len(eye_detected_images)} images "
        f"contained YOLO eye detections, "
        f"but {NUM_IMAGES} are required."
    )


# ============================================================
# SELECT 20 DRIVER IMAGES
# ============================================================

random.seed(RANDOM_SEED)

selected_images = random.sample(
    eye_detected_images,
    NUM_IMAGES
)

print(
    f"\nSelected {NUM_IMAGES} images "
    f"using seed {RANDOM_SEED}."
)


# ============================================================
# SAVE YOLO-SELECTED IMAGES
# ============================================================

print(
    "\nSaving YOLO-selected images..."
)

for index, image_path in enumerate(
    selected_images,
    start=1
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        continue

    output_path = (
        YOLO_DETECTED_DIR
        / f"{index:02d}_{image_path.name}"
    )

    cv2.imwrite(
        str(output_path),
        image
    )


# ============================================================
# LOAD MEDIAPIPE
# ============================================================

print("\n" + "=" * 70)
print("3. LOADING MEDIAPIPE FACE LANDMARKER")
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

print(
    "MediaPipe loaded successfully."
)


# ============================================================
# RUN MEDIAPIPE
# ============================================================

print("\n" + "=" * 70)
print("4. TESTING MEDIAPIPE ON DRIVER IMAGES")
print("=" * 70)


mediapipe_detected = []

mediapipe_failed = []

read_failures = []

runtime_failures = []


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

        print(
            "  Status: IMAGE READ FAILURE"
        )

        read_failures.append(
            image_path.name
        )

        continue


    height, width = image.shape[:2]


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
            "  Status: MEDIAPIPE RUNTIME ERROR"
        )

        print(
            f"  Error: {e}"
        )

        runtime_failures.append(
            image_path.name
        )

        continue


    # ========================================================
    # FACE NOT DETECTED
    # ========================================================

    if not detection_result.face_landmarks:

        print(
            "  Status: FACE NOT DETECTED"
        )

        mediapipe_failed.append(
            image_path.name
        )

        # Save original image.
        output_path = (
            MEDIAPIPE_FAILED_DIR
            / f"{index:02d}_{image_path.name}"
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        continue


    # ========================================================
    # FACE DETECTED
    # ========================================================

    print(
        "  Status: FACE DETECTED"
    )

    landmarks = (
        detection_result.face_landmarks[0]
    )

    print(
        f"  Landmarks: {len(landmarks)}"
    )

    mediapipe_detected.append(
        image_path.name
    )


    # ========================================================
    # ANNOTATE
    # ========================================================

    annotated = image.copy()


    # --------------------------------------------------------
    # Draw landmarks
    # --------------------------------------------------------

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
    # Save detected image
    # --------------------------------------------------------

    output_path = (
        MEDIAPIPE_DETECTED_DIR
        / f"{index:02d}_{image_path.name}"
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
print("5. FINAL RESULTS")
print("=" * 70)

print(
    f"Driver-related images selected : "
    f"{NUM_IMAGES}"
)

print(
    f"MediaPipe faces detected        : "
    f"{len(mediapipe_detected)}"
)

print(
    f"MediaPipe face failures         : "
    f"{len(mediapipe_failed)}"
)

print(
    f"Image read failures             : "
    f"{len(read_failures)}"
)

print(
    f"MediaPipe runtime failures      : "
    f"{len(runtime_failures)}"
)


# ============================================================
# DETECTION RATE
# ============================================================

if NUM_IMAGES > 0:

    detection_rate = (
        100
        * len(mediapipe_detected)
        / NUM_IMAGES
    )

    print(
        f"\nMediaPipe detection rate       : "
        f"{detection_rate:.1f}%"
    )


# ============================================================
# FAILED IMAGES
# ============================================================

print("\n" + "=" * 70)
print("MEDIAPIPE FAILED IMAGES")
print("=" * 70)

if mediapipe_failed:

    for index, filename in enumerate(
        mediapipe_failed,
        start=1
    ):

        print(
            f"{index:02d}. {filename}"
        )

else:

    print(
        "No MediaPipe face detection failures."
    )


# ============================================================
# DETECTED IMAGES
# ============================================================

print("\n" + "=" * 70)
print("MEDIAPIPE DETECTED IMAGES")
print("=" * 70)

if mediapipe_detected:

    for index, filename in enumerate(
        mediapipe_detected,
        start=1
    ):

        print(
            f"{index:02d}. {filename}"
        )

else:

    print(
        "No faces were detected."
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT DIRECTORIES")
print("=" * 70)

print(
    f"YOLO-selected images:\n"
    f"{YOLO_DETECTED_DIR}"
)

print(
    f"\nMediaPipe detected:\n"
    f"{MEDIAPIPE_DETECTED_DIR}"
)

print(
    f"\nMediaPipe failed:\n"
    f"{MEDIAPIPE_FAILED_DIR}"
)

print("\n" + "=" * 70)
print(
    "MEDIAPIPE DRIVER IMAGE TEST COMPLETED"
)
print("=" * 70)