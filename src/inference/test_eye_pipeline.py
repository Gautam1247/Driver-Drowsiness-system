from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

YOLO_MODEL = PROJECT_ROOT / "models" / "dms_yolov8n_best.pt"
EYE_MODEL = PROJECT_ROOT / "models" / "eye_mobilenetv2_baseline_best.pth"

TEST_IMAGE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "yolo_dms"
    / "images"
    / "test"
)

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "eye_pipeline"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

YOLO_CONFIDENCE = 0.40
EYE_CONFIDENCE = 0.50

IMAGE_SIZE = 224


# ============================================================
# EYE CNN TRANSFORM
# ============================================================

eye_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# LOAD YOLO
# ============================================================

print("=" * 60)
print("YOLO + EYE CNN PIPELINE TEST")
print("=" * 60)

print("\nLoading YOLO...")

yolo_model = YOLO(str(YOLO_MODEL))

print("YOLO loaded.")
print("Classes:", yolo_model.names)


# ============================================================
# LOAD EYE CNN
# ============================================================

print("\nLoading Eye CNN...")

eye_model = models.mobilenet_v2(weights=None)

num_features = eye_model.classifier[1].in_features

eye_model.classifier[1] = nn.Linear(
    num_features,
    2
)

checkpoint = torch.load(
    EYE_MODEL,
    map_location=DEVICE
)

# Our saved model is a checkpoint dictionary.
if "model_state_dict" in checkpoint:
    eye_state_dict = checkpoint["model_state_dict"]
else:
    eye_state_dict = checkpoint

eye_model.load_state_dict(eye_state_dict)

eye_model = eye_model.to(DEVICE)
eye_model.eval()

print("Eye CNN loaded.")
print("Classes:")
print("  0 = CLOSED")
print("  1 = OPEN")


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

annotated = image.copy()


# ============================================================
# YOLO INFERENCE
# ============================================================

print("\nRunning YOLO...")

results = yolo_model.predict(
    source=str(image_path),
    conf=YOLO_CONFIDENCE,
    device="cpu",
    verbose=False,
)

result = results[0]


# ============================================================
# PROCESS EYE DETECTIONS
# ============================================================

eye_count = 0

if result.boxes is not None:

    for box in result.boxes:

        class_id = int(box.cls[0])
        yolo_conf = float(box.conf[0])

        class_name = yolo_model.names[class_id]

        # Only process YOLO eye detections.
        if class_name not in {"open_eye", "closed_eye"}:
            continue

        if yolo_conf < YOLO_CONFIDENCE:
            continue

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image.shape[1], int(x2))
        y2 = min(image.shape[0], int(y2))

        if x2 <= x1 or y2 <= y1:
            continue

        eye_crop = image[y1:y2, x1:x2]

        if eye_crop.size == 0:
            continue

        eye_count += 1

        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        eye_rgb = cv2.cvtColor(
            eye_crop,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Prepare CNN input
        # ----------------------------------------------------

        tensor = eye_transform(eye_rgb)

        tensor = tensor.unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # Eye CNN inference
        # ----------------------------------------------------

        with torch.no_grad():

            output = eye_model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

            predicted_class = int(
                torch.argmax(probabilities)
            )

            cnn_conf = float(
                probabilities[predicted_class]
            )

        if predicted_class == 0:
            eye_state = "CLOSED"
        else:
            eye_state = "OPEN"

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"\nEye {eye_count}:"
        )

        print(
            f"  YOLO class      : {class_name}"
        )

        print(
            f"  YOLO confidence : {yolo_conf:.3f}"
        )

        print(
            f"  CNN prediction  : {eye_state}"
        )

        print(
            f"  CNN confidence  : {cnn_conf:.3f}"
        )

        print(
            f"  Box             : "
            f"({x1}, {y1}, {x2}, {y2})"
        )

        # ----------------------------------------------------
        # Draw result
        # ----------------------------------------------------

        label = (
            f"{eye_state} "
            f"{cnn_conf:.2f}"
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


# ============================================================
# SAVE RESULT
# ============================================================

output_path = (
    OUTPUT_DIR
    / f"eye_pipeline_{image_path.name}"
)

cv2.imwrite(
    str(output_path),
    annotated
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)

print(
    f"Eye detections processed: {eye_count}"
)

print(
    f"Annotated image saved:"
)

print(output_path)

print("=" * 60)
print("YOLO + EYE CNN PIPELINE COMPLETED")
print("=" * 60)