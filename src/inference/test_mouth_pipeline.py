from pathlib import Path

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
    / "mouth_pipeline"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

IMAGE_SIZE = 224

# Mouth crop padding.
# These values match the strategy used when creating
# the mouth CNN dataset.
X_PADDING = 0.30
Y_PADDING = 0.45


# ============================================================
# MOUTH CNN TRANSFORM
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
# LOAD MOUTH CNN
# ============================================================

print("=" * 60)
print("MEDIAPIPE + MOUTH CNN PIPELINE TEST")
print("=" * 60)

print("\nLoading Mouth CNN...")

mouth_model = models.mobilenet_v2(weights=None)

num_features = mouth_model.classifier[1].in_features

mouth_model.classifier[1] = nn.Linear(
    num_features,
    2
)

checkpoint = torch.load(
    MOUTH_MODEL,
    map_location=DEVICE
)

# Support both checkpoint dictionaries and raw state_dicts.
if "model_state_dict" in checkpoint:
    mouth_state_dict = checkpoint["model_state_dict"]
else:
    mouth_state_dict = checkpoint

mouth_model.load_state_dict(mouth_state_dict)

mouth_model = mouth_model.to(DEVICE)
mouth_model.eval()

print("Mouth CNN loaded.")

print("Classes:")
print("  0 = NON_YAWNING")
print("  1 = YAWNING")


# ============================================================
# LOAD MEDIAPIPE FACE LANDMARKER
# ============================================================

print("\nLoading MediaPipe Face Landmarker...")

base_options = python.BaseOptions(
    model_asset_path=str(FACE_MODEL)
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    options
)

print("MediaPipe Face Landmarker loaded.")


# ============================================================
# FIND TEST IMAGE
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

images = sorted(
    p
    for p in TEST_IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in image_extensions
)

if not images:
    raise RuntimeError("No test images found.")

image_path = images[0]

print("\nSelected image:")
print(image_path)


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError(
        f"Could not read image: {image_path}"
    )

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# MEDIAPIPE LANDMARK DETECTION
# ============================================================

print("\nRunning MediaPipe...")

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=image_rgb
)

detection_result = face_landmarker.detect(
    mp_image
)

if not detection_result.face_landmarks:
    raise RuntimeError(
        "MediaPipe did not detect a face."
    )

landmarks = detection_result.face_landmarks[0]

print(
    f"Face detected successfully."
)

print(
    f"Landmarks detected: {len(landmarks)}"
)


# ============================================================
# MOUTH LANDMARKS
# ============================================================

# MediaPipe Face Mesh / Face Landmarker lip landmarks.
# We use the outer mouth contour to construct the ROI.

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
# CONVERT LANDMARKS TO PIXELS
# ============================================================

height, width = image.shape[:2]

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


# ============================================================
# CREATE MOUTH ROI
# ============================================================

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

mouth_width = max_x - min_x
mouth_height = max_y - min_y

# Add padding around the mouth.
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

if x2 <= x1 or y2 <= y1:
    raise RuntimeError(
        "Invalid mouth bounding box."
    )


mouth_crop = image[
    y1:y2,
    x1:x2
]

if mouth_crop.size == 0:
    raise RuntimeError(
        "Mouth crop is empty."
    )


print("\nMouth ROI:")
print(
    f"  Box: ({x1}, {y1}, {x2}, {y2})"
)

print(
    f"  Size: {mouth_crop.shape[1]} x "
    f"{mouth_crop.shape[0]}"
)


# ============================================================
# MOUTH CNN INFERENCE
# ============================================================

print("\nRunning Mouth CNN...")

mouth_rgb = cv2.cvtColor(
    mouth_crop,
    cv2.COLOR_BGR2RGB
)

tensor = mouth_transform(
    mouth_rgb
)

tensor = tensor.unsqueeze(0).to(
    DEVICE
)

with torch.no_grad():

    output = mouth_model(
        tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )[0]

    predicted_class = int(
        torch.argmax(probabilities)
    )

    cnn_confidence = float(
        probabilities[predicted_class]
    )


if predicted_class == 0:
    mouth_state = "NON-YAWNING"
else:
    mouth_state = "YAWNING"


print("\nMouth prediction:")
print(
    f"  Prediction  : {mouth_state}"
)

print(
    f"  Confidence  : {cnn_confidence:.3f}"
)

print(
    f"  Non-yawning : "
    f"{float(probabilities[0]):.3f}"
)

print(
    f"  Yawning     : "
    f"{float(probabilities[1]):.3f}"
)


# ============================================================
# DRAW RESULT
# ============================================================

annotated = image.copy()

cv2.rectangle(
    annotated,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    2
)

label = (
    f"{mouth_state} "
    f"{cnn_confidence:.2f}"
)

cv2.putText(
    annotated,
    label,
    (x1, max(y1 - 10, 20)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 255, 0),
    2,
    cv2.LINE_AA
)


# ============================================================
# DRAW LANDMARKS
# ============================================================

for x, y in mouth_points:

    cv2.circle(
        annotated,
        (x, y),
        1,
        (255, 0, 0),
        -1
    )


# ============================================================
# SAVE OUTPUT
# ============================================================

output_path = (
    OUTPUT_DIR
    / f"mouth_pipeline_{image_path.name}"
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

print("\n" + "=" * 60)

print(
    "Mouth pipeline completed successfully."
)

print(
    f"Result image saved:"
)

print(output_path)

print("=" * 60)