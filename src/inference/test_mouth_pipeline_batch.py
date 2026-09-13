from pathlib import Path
import random
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MOUTH_MODEL = (
    PROJECT_ROOT
    / "models"
    / "mouth_mobilenetv2_yawning_best.pth"
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
    / "mouth_pipeline_batch"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_PATH = OUTPUT_DIR / "mouth_batch_results.csv"


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

IMAGE_SIZE = 224

NUM_IMAGES = 20

RANDOM_SEED = 42

X_PADDING = 0.30
Y_PADDING = 0.45


# ============================================================
# TRANSFORM
# ============================================================

mouth_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BATCH TEST - MEDIAPIPE + MOUTH CNN")
print("=" * 70)

print(f"Project root : {PROJECT_ROOT}")
print(f"Test images  : {TEST_IMAGE_DIR}")
print(f"Output       : {OUTPUT_DIR}")
print(f"Images       : {NUM_IMAGES}")
print(f"Random seed  : {RANDOM_SEED}")
print(f"Device       : {DEVICE}")


# ============================================================
# CHECK FILES
# ============================================================

if not MOUTH_MODEL.exists():
    raise FileNotFoundError(
        f"Mouth model not found:\n{MOUTH_MODEL}"
    )

if not FACE_MODEL.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found:\n{FACE_MODEL}"
    )

if not TEST_IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Test image directory not found:\n{TEST_IMAGE_DIR}"
    )


# ============================================================
# LOAD MOUTH CNN
# ============================================================

print("\n" + "=" * 70)
print("1. LOADING MOUTH CNN")
print("=" * 70)

mouth_model = models.mobilenet_v2(
    weights=None
)

num_features = (
    mouth_model.classifier[1].in_features
)

mouth_model.classifier[1] = nn.Linear(
    num_features,
    2
)

checkpoint = torch.load(
    MOUTH_MODEL,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

mouth_model.load_state_dict(
    state_dict
)

mouth_model = mouth_model.to(
    DEVICE
)

mouth_model.eval()

print("Mouth CNN loaded successfully.")
print("Classes:")
print("  0 = NON_YAWNING")
print("  1 = YAWNING")


# ============================================================
# LOAD MEDIAPIPE
# ============================================================

print("\n" + "=" * 70)
print("2. LOADING MEDIAPIPE FACE LANDMARKER")
print("=" * 70)

base_options = python.BaseOptions(
    model_asset_path=str(FACE_MODEL)
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.30,
    min_face_presence_confidence=0.30,
)

face_landmarker = (
    vision.FaceLandmarker.create_from_options(
        options
    )
)

print(
    "MediaPipe Face Landmarker loaded successfully."
)


# ============================================================
# FIND TEST IMAGES
# ============================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

all_images = sorted(
    p
    for p in TEST_IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in extensions
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
# RANDOM SAMPLE
# ============================================================

random.seed(RANDOM_SEED)

selected_images = random.sample(
    all_images,
    NUM_IMAGES
)

print(
    f"Selected {len(selected_images)} images "
    f"for batch testing."
)


# ============================================================
# MOUTH LANDMARK INDICES
# ============================================================

MOUTH_INDICES = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
    76,
    62,
    96,
    89,
    72,
    11,
    302,
    318,
    324,
    308,
    415,
    310,
    311,
    312,
    13,
    82,
    81,
    80,
    191,
    78,
]


# ============================================================
# PROCESS IMAGES
# ============================================================

results = []

successful = 0
face_failures = 0
read_failures = 0
inference_failures = 0

yawning_count = 0
non_yawning_count = 0


print("\n" + "=" * 70)
print("3. RUNNING BATCH INFERENCE")
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

        print("  ERROR: Could not read image.")

        read_failures += 1

        results.append({
            "image": image_path.name,
            "status": "READ_FAILURE",
            "prediction": "",
            "confidence": "",
            "non_yawning_probability": "",
            "yawning_probability": "",
            "mouth_x1": "",
            "mouth_y1": "",
            "mouth_x2": "",
            "mouth_y2": "",
        })

        continue

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
            f"  ERROR: MediaPipe failed: {e}"
        )

        inference_failures += 1

        results.append({
            "image": image_path.name,
            "status": "MEDIAPIPE_FAILURE",
            "prediction": "",
            "confidence": "",
            "non_yawning_probability": "",
            "yawning_probability": "",
            "mouth_x1": "",
            "mouth_y1": "",
            "mouth_x2": "",
            "mouth_y2": "",
        })

        continue


    # --------------------------------------------------------
    # Check face
    # --------------------------------------------------------

    if not detection_result.face_landmarks:

        print("  Face: NOT DETECTED")

        face_failures += 1

        results.append({
            "image": image_path.name,
            "status": "NO_FACE",
            "prediction": "",
            "confidence": "",
            "non_yawning_probability": "",
            "yawning_probability": "",
            "mouth_x1": "",
            "mouth_y1": "",
            "mouth_x2": "",
            "mouth_y2": "",
        })

        continue


    landmarks = (
        detection_result.face_landmarks[0]
    )


    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    height, width = image.shape[:2]


    # --------------------------------------------------------
    # Convert mouth landmarks to pixels
    # --------------------------------------------------------

    mouth_points = []

    for idx in MOUTH_INDICES:

        landmark = landmarks[idx]

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        x = max(
            0,
            min(width - 1, x)
        )

        y = max(
            0,
            min(height - 1, y)
        )

        mouth_points.append(
            (x, y)
        )


    mouth_points = np.array(
        mouth_points
    )


    # --------------------------------------------------------
    # Mouth bounding box
    # --------------------------------------------------------

    min_x = int(
        mouth_points[:, 0].min()
    )

    max_x = int(
        mouth_points[:, 0].max()
    )

    min_y = int(
        mouth_points[:, 1].min()
    )

    max_y = int(
        mouth_points[:, 1].max()
    )

    mouth_width = (
        max_x - min_x
    )

    mouth_height = (
        max_y - min_y
    )

    if mouth_width <= 0 or mouth_height <= 0:

        print(
            "  ERROR: Invalid mouth ROI."
        )

        inference_failures += 1

        results.append({
            "image": image_path.name,
            "status": "INVALID_ROI",
            "prediction": "",
            "confidence": "",
            "non_yawning_probability": "",
            "yawning_probability": "",
            "mouth_x1": "",
            "mouth_y1": "",
            "mouth_x2": "",
            "mouth_y2": "",
        })

        continue


    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    pad_x = int(
        mouth_width * X_PADDING
    )

    pad_y = int(
        mouth_height * Y_PADDING
    )

    x1 = max(
        0,
        min_x - pad_x
    )

    y1 = max(
        0,
        min_y - pad_y
    )

    x2 = min(
        width,
        max_x + pad_x
    )

    y2 = min(
        height,
        max_y + pad_y
    )


    # --------------------------------------------------------
    # Crop mouth
    # --------------------------------------------------------

    mouth_crop = image[
        y1:y2,
        x1:x2
    ]

    if mouth_crop.size == 0:

        print(
            "  ERROR: Empty mouth crop."
        )

        inference_failures += 1

        results.append({
            "image": image_path.name,
            "status": "EMPTY_CROP",
            "prediction": "",
            "confidence": "",
            "non_yawning_probability": "",
            "yawning_probability": "",
            "mouth_x1": x1,
            "mouth_y1": y1,
            "mouth_x2": x2,
            "mouth_y2": y2,
        })

        continue


    # --------------------------------------------------------
    # CNN input
    # --------------------------------------------------------

    mouth_rgb = cv2.cvtColor(
        mouth_crop,
        cv2.COLOR_BGR2RGB
    )

    tensor = mouth_transform(
        mouth_rgb
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .to(DEVICE)
    )


    # --------------------------------------------------------
    # Mouth CNN
    # --------------------------------------------------------

    try:

        with torch.no_grad():

            output = mouth_model(
                tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

            predicted_class = int(
                torch.argmax(
                    probabilities
                )
            )

            confidence = float(
                probabilities[
                    predicted_class
                ]
            )

            non_yawning_probability = float(
                probabilities[0]
            )

            yawning_probability = float(
                probabilities[1]
            )

    except Exception as e:

        print(
            f"  ERROR: CNN inference failed: {e}"
        )

        inference_failures += 1

        results.append({
            "image": image_path.name,
            "status": "CNN_FAILURE",
            "prediction": "",
            "confidence": "",
            "non_yawning_probability": "",
            "yawning_probability": "",
            "mouth_x1": x1,
            "mouth_y1": y1,
            "mouth_x2": x2,
            "mouth_y2": y2,
        })

        continue


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if predicted_class == 1:

        prediction = "YAWNING"

        yawning_count += 1

    else:

        prediction = "NON_YAWNING"

        non_yawning_count += 1


    successful += 1


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"  Face       : DETECTED"
    )

    print(
        f"  Mouth ROI  : "
        f"({x1}, {y1}, {x2}, {y2})"
    )

    print(
        f"  Prediction : "
        f"{prediction}"
    )

    print(
        f"  Confidence : "
        f"{confidence:.3f}"
    )

    print(
        f"  Non-yawn   : "
        f"{non_yawning_probability:.3f}"
    )

    print(
        f"  Yawning    : "
        f"{yawning_probability:.3f}"
    )


    # --------------------------------------------------------
    # Annotate image
    # --------------------------------------------------------

    annotated = image.copy()

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = (
        f"{prediction} "
        f"{confidence:.2f}"
    )

    cv2.putText(
        annotated,
        label,
        (
            x1,
            max(y1 - 10, 20)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    # Draw mouth landmarks.

    for x, y in mouth_points:

        cv2.circle(
            annotated,
            (x, y),
            1,
            (255, 0, 0),
            -1
        )


    # --------------------------------------------------------
    # Save annotated image
    # --------------------------------------------------------

    output_image = (
        OUTPUT_DIR
        / f"{index:02d}_{image_path.name}"
    )

    cv2.imwrite(
        str(output_image),
        annotated
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({
        "image": image_path.name,
        "status": "SUCCESS",
        "prediction": prediction,
        "confidence": f"{confidence:.6f}",
        "non_yawning_probability": (
            f"{non_yawning_probability:.6f}"
        ),
        "yawning_probability": (
            f"{yawning_probability:.6f}"
        ),
        "mouth_x1": x1,
        "mouth_y1": y1,
        "mouth_x2": x2,
        "mouth_y2": y2,
    })


# ============================================================
# SAVE CSV
# ============================================================

print("\n" + "=" * 70)
print("4. SAVING RESULTS")
print("=" * 70)

fieldnames = [
    "image",
    "status",
    "prediction",
    "confidence",
    "non_yawning_probability",
    "yawning_probability",
    "mouth_x1",
    "mouth_y1",
    "mouth_x2",
    "mouth_y2",
]

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("5. BATCH TEST SUMMARY")
print("=" * 70)

print(
    f"Images selected       : {NUM_IMAGES}"
)

print(
    f"Successful             : {successful}"
)

print(
    f"Face detection failure : {face_failures}"
)

print(
    f"Image read failures    : {read_failures}"
)

print(
    f"Inference failures     : {inference_failures}"
)

print(
    f"Yawning predictions    : {yawning_count}"
)

print(
    f"Non-yawning predictions: {non_yawning_count}"
)

if successful > 0:

    print(
        f"Yawning percentage     : "
        f"{100 * yawning_count / successful:.1f}%"
    )

    print(
        f"Non-yawning percentage : "
        f"{100 * non_yawning_count / successful:.1f}%"
    )


print("\nResults CSV:")
print(CSV_PATH)

print("\nAnnotated images:")
print(OUTPUT_DIR)

print("\n" + "=" * 70)
print("BATCH MOUTH PIPELINE TEST COMPLETED")
print("=" * 70)


# ============================================================
# CLEANUP
# ============================================================

face_landmarker.close()