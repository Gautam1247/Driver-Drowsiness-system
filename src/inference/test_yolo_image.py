from pathlib import Path
from ultralytics import YOLO
import cv2


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "dms_yolov8n_best.pt"
TEST_IMAGE_DIR = PROJECT_ROOT / "datasets" / "yolo_dms" / "images" / "test"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "yolo_test"


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE = 0.40


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("YOLO SINGLE IMAGE TEST")
print("=" * 60)

print(f"Project root : {PROJECT_ROOT}")
print(f"Model        : {MODEL_PATH}")
print(f"Test images  : {TEST_IMAGE_DIR}")
print(f"Output       : {OUTPUT_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

if not TEST_IMAGE_DIR.exists():
    raise FileNotFoundError(f"Test image directory not found: {TEST_IMAGE_DIR}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("YOLO loaded successfully.")
print("Classes:", model.names)


# ============================================================
# SELECT ONE TEST IMAGE
# ============================================================

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

images = sorted(
    p for p in TEST_IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
)

if not images:
    raise RuntimeError("No test images found.")

image_path = images[0]

print(f"\nSelected image: {image_path.name}")


# ============================================================
# RUN INFERENCE
# ============================================================

print("\nRunning inference...")

results = model.predict(
    source=str(image_path),
    conf=CONFIDENCE,
    device="cpu",
    verbose=False,
)


# ============================================================
# DISPLAY DETECTIONS IN TERMINAL
# ============================================================

result = results[0]

print("\nDetections:")

if result.boxes is None or len(result.boxes) == 0:
    print("No objects detected.")
else:
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = model.names[class_id]

        print(
            f"{i + 1}. "
            f"{class_name:12s} "
            f"confidence={confidence:.3f} "
            f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )


# ============================================================
# SAVE ANNOTATED IMAGE
# ============================================================

annotated = result.plot()

output_path = OUTPUT_DIR / f"yolo_result_{image_path.name}"

cv2.imwrite(
    str(output_path),
    annotated
)

print("\nAnnotated image saved:")
print(output_path)

print("\n" + "=" * 60)
print("YOLO IMAGE TEST COMPLETED")
print("=" * 60)