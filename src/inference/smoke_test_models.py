from pathlib import Path

import torch
from torchvision import models
import torch.nn as nn
from ultralytics import YOLO
import mediapipe as mp


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

YOLO_MODEL = PROJECT_ROOT / "models" / "dms_yolov8n_best.pt"
EYE_MODEL = PROJECT_ROOT / "models" / "eye_mobilenetv2_baseline_best.pth"
MOUTH_MODEL = PROJECT_ROOT / "models" / "mouth_mobilenetv2_yawning_best.pth"
FACE_MODEL = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("=" * 60)
print("DRIVER MONITORING SYSTEM - MODEL SMOKE TEST")
print("=" * 60)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Device       : {device}")

print("\nModel files:")

for name, path in [
    ("YOLO", YOLO_MODEL),
    ("Eye CNN", EYE_MODEL),
    ("Mouth CNN", MOUTH_MODEL),
    ("MediaPipe", FACE_MODEL),
]:
    print(f"{name:12}: {path.exists()} -> {path}")


# ============================================================
# 1. YOLO
# ============================================================

print("\n" + "=" * 60)
print("1. Loading YOLO")
print("=" * 60)

yolo_model = YOLO(str(YOLO_MODEL))

print("YOLO loaded successfully.")
print("Classes:", yolo_model.names)


# ============================================================
# 2. EYE CNN
# ============================================================

print("\n" + "=" * 60)
print("2. Loading Eye CNN")
print("=" * 60)

eye_model = models.mobilenet_v2(weights=None)

num_features = eye_model.classifier[1].in_features
eye_model.classifier[1] = nn.Linear(num_features, 2)

eye_checkpoint = torch.load(
    EYE_MODEL,
    map_location=device
)

if "model_state_dict" in eye_checkpoint:
    eye_state_dict = eye_checkpoint["model_state_dict"]
else:
    eye_state_dict = eye_checkpoint

eye_model.load_state_dict(eye_state_dict)

eye_model = eye_model.to(device)
eye_model.eval()

print("Eye CNN loaded successfully.")
print("Classes: CLOSED = 0, OPEN = 1")


# ============================================================
# 3. MOUTH CNN
# ============================================================

print("\n" + "=" * 60)
print("3. Loading Mouth CNN")
print("=" * 60)

mouth_model = models.mobilenet_v2(weights=None)

num_features = mouth_model.classifier[1].in_features
mouth_model.classifier[1] = nn.Linear(num_features, 2)
mouth_checkpoint = torch.load(
    MOUTH_MODEL,
    map_location=device
)

if "model_state_dict" in mouth_checkpoint:
    mouth_state_dict = mouth_checkpoint["model_state_dict"]
else:
    mouth_state_dict = mouth_checkpoint

mouth_model.load_state_dict(mouth_state_dict)

mouth_model = mouth_model.to(device)
mouth_model.eval()

print("Mouth CNN loaded successfully.")
print("Classes: NON_YAWNING = 0, YAWNING = 1")


# ============================================================
# 4. MEDIAPIPE FACE LANDMARKER
# ============================================================

print("\n" + "=" * 60)
print("4. Loading MediaPipe Face Landmarker")
print("=" * 60)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(FACE_MODEL)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

face_landmarker = FaceLandmarker.create_from_options(options)

print("MediaPipe Face Landmarker loaded successfully.")


# ============================================================
# FINAL RESULT
# ============================================================

face_landmarker.close()

print("\n" + "=" * 60)
print("ALL MODELS LOADED SUCCESSFULLY")
print("=" * 60)
print("\nReady to begin real-time inference integration.")