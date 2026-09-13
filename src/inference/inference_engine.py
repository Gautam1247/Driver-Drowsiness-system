from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class DriverInferenceEngine:
    """
    Unified single-frame inference engine.

    Pipeline:

        Frame
          |
          +--> YOLO
          |      |
          |      +--> Eyes --> Eye CNN
          |      +--> Phone
          |      +--> Seatbelt
          |
          +--> MediaPipe --> Mouth ROI --> Mouth CNN

    Temporal reasoning is intentionally NOT included here.
    """


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        project_root=None,
        device="cpu",
        yolo_confidence=0.25,
        face_detection_confidence=0.30,
        face_presence_confidence=0.30,
    ):

        # ----------------------------------------------------
        # Project root
        # ----------------------------------------------------

        if project_root is None:

            project_root = (
                Path(__file__)
                .resolve()
                .parents[2]
            )

        self.project_root = Path(
            project_root
        )

        self.device = torch.device(
            device
        )


        # ----------------------------------------------------
        # Model paths
        # ----------------------------------------------------

        self.yolo_path = (
            self.project_root
            / "models"
            / "dms_yolov8n_best.pt"
        )

        self.eye_path = (
            self.project_root
            / "models"
            / "eye_mobilenetv2_baseline_best.pth"
        )

        self.mouth_path = (
            self.project_root
            / "models"
            / "mouth_mobilenetv2_yawning_best.pth"
        )

        self.face_path = (
            self.project_root
            / "models"
            / "mediapipe"
            / "face_landmarker.task"
        )


        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        self.yolo_confidence = (
            yolo_confidence
        )

        self.face_detection_confidence = (
            face_detection_confidence
        )

        self.face_presence_confidence = (
            face_presence_confidence
        )

        self.image_size = 224

        # CPU live-inference tuning. These do not change model weights.
        # A smaller YOLO inference size and capped detections reduce CPU
        # latency; mouth inference is refreshed every second frame because
        # yawning is a temporal event rather than a per-pixel frame task.
        self.yolo_image_size = 448
        self.yolo_max_detections = 20
        self.mouth_inference_interval = 2
        self._frame_counter = 0
        self._last_mouth_result = None

        self.x_padding = 0.30
        self.y_padding = 0.45


        # ----------------------------------------------------
        # Expected YOLO classes
        # ----------------------------------------------------

        self.OPEN_EYE_CLASS = 0
        self.CLOSED_EYE_CLASS = 1
        self.PHONE_CLASS = 2
        self.SEATBELT_CLASS = 3


        # ----------------------------------------------------
        # Mouth classes
        # ----------------------------------------------------

        self.NON_YAWNING_CLASS = 0
        self.YAWNING_CLASS = 1


        # ----------------------------------------------------
        # MediaPipe mouth landmarks
        # ----------------------------------------------------

        self.MOUTH_INDICES = [
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


        # ----------------------------------------------------
        # CNN preprocessing
        # ----------------------------------------------------

        self.cnn_transform = (
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(
                    (
                        self.image_size,
                        self.image_size
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225
                    ]
                ),
            ])
        )


        # ----------------------------------------------------
        # Load models
        # ----------------------------------------------------

        print("Loading YOLO...")
        self._load_yolo()

        print("Loading Eye CNN...")
        self._load_eye_model()

        print("Loading Mouth CNN...")
        self._load_mouth_model()

        print("Loading MediaPipe...")
        self._load_mediapipe()

        print("All models loaded successfully.")


    # ========================================================
    # LOAD YOLO
    # ========================================================

    def _load_yolo(self):

        if not self.yolo_path.exists():

            raise FileNotFoundError(
                f"YOLO model not found:\n"
                f"{self.yolo_path}"
            )

        self.yolo_model = YOLO(
            str(self.yolo_path)
        )

        print(
            f"  YOLO classes: "
            f"{self.yolo_model.names}"
        )


    # ========================================================
    # LOAD EYE CNN
    # ========================================================

    def _load_eye_model(self):

        if not self.eye_path.exists():

            raise FileNotFoundError(
                f"Eye model not found:\n"
                f"{self.eye_path}"
            )

        self.eye_model = (
            models.mobilenet_v2(
                weights=None
            )
        )

        num_features = (
            self.eye_model
            .classifier[1]
            .in_features
        )

        self.eye_model.classifier[1] = (
            nn.Linear(
                num_features,
                2
            )
        )

        checkpoint = torch.load(
            self.eye_path,
            map_location=self.device
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            state_dict = checkpoint

        self.eye_model.load_state_dict(
            state_dict
        )

        self.eye_model = (
            self.eye_model
            .to(self.device)
        )

        self.eye_model.eval()

        print(
            "  Eye classes:"
        )

        print(
            "    0 = CLOSED"
        )

        print(
            "    1 = OPEN"
        )


    # ========================================================
    # LOAD MOUTH CNN
    # ========================================================

    def _load_mouth_model(self):

        if not self.mouth_path.exists():

            raise FileNotFoundError(
                f"Mouth model not found:\n"
                f"{self.mouth_path}"
            )

        self.mouth_model = (
            models.mobilenet_v2(
                weights=None
            )
        )

        num_features = (
            self.mouth_model
            .classifier[1]
            .in_features
        )

        self.mouth_model.classifier[1] = (
            nn.Linear(
                num_features,
                2
            )
        )

        checkpoint = torch.load(
            self.mouth_path,
            map_location=self.device
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            state_dict = checkpoint

        self.mouth_model.load_state_dict(
            state_dict
        )

        self.mouth_model = (
            self.mouth_model
            .to(self.device)
        )

        self.mouth_model.eval()

        print(
            "  Mouth classes:"
        )

        print(
            "    0 = NON_YAWNING"
        )

        print(
            "    1 = YAWNING"
        )


    # ========================================================
    # LOAD MEDIAPIPE
    # ========================================================

    def _load_mediapipe(self):

        if not self.face_path.exists():

            raise FileNotFoundError(
                f"MediaPipe model not found:\n"
                f"{self.face_path}"
            )

        base_options = (
            python.BaseOptions(
                model_asset_path=str(
                    self.face_path
                )
            )
        )

        options = (
            vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=(
                    vision.RunningMode.IMAGE
                ),
                num_faces=1,
                min_face_detection_confidence=(
                    self.face_detection_confidence
                ),
                min_face_presence_confidence=(
                    self.face_presence_confidence
                ),
            )
        )

        self.face_landmarker = (
            vision.FaceLandmarker
            .create_from_options(
                options
            )
        )


    # ========================================================
    # EYE CNN PREDICTION
    # ========================================================

    def _predict_eye(
        self,
        eye_crop
    ):

        eye_rgb = cv2.cvtColor(
            eye_crop,
            cv2.COLOR_BGR2RGB
        )

        tensor = (
            self.cnn_transform(
                eye_rgb
            )
        )

        tensor = (
            tensor
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.inference_mode():

            output = (
                self.eye_model(
                    tensor
                )
            )

            probabilities = (
                torch.softmax(
                    output,
                    dim=1
                )[0]
            )

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

        if predicted_class == 0:

            state = "CLOSED"

        else:

            state = "OPEN"

        return {
            "state": state,
            "class_id": predicted_class,
            "confidence": confidence,
            "closed_probability": float(
                probabilities[0]
            ),
            "open_probability": float(
                probabilities[1]
            ),
        }


    # ========================================================
    # MOUTH CNN PREDICTION
    # ========================================================

    def _predict_mouth(
        self,
        mouth_crop
    ):

        mouth_rgb = cv2.cvtColor(
            mouth_crop,
            cv2.COLOR_BGR2RGB
        )

        tensor = (
            self.cnn_transform(
                mouth_rgb
            )
        )

        tensor = (
            tensor
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.inference_mode():

            output = (
                self.mouth_model(
                    tensor
                )
            )

            probabilities = (
                torch.softmax(
                    output,
                    dim=1
                )[0]
            )

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

        if predicted_class == 1:

            state = "YAWNING"

        else:

            state = "NON_YAWNING"

        return {
            "state": state,
            "class_id": predicted_class,
            "confidence": confidence,
            "non_yawning_probability": float(
                probabilities[0]
            ),
            "yawning_probability": float(
                probabilities[1]
            ),
        }


    # ========================================================
    # CREATE MOUTH ROI
    # ========================================================

    def _get_mouth_roi(
        self,
        image,
        landmarks
    ):

        height, width = (
            image.shape[:2]
        )

        mouth_points = []

        for idx in self.MOUTH_INDICES:

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

        if (
            mouth_width <= 0
            or mouth_height <= 0
        ):

            return None

        pad_x = int(
            mouth_width
            * self.x_padding
        )

        pad_y = int(
            mouth_height
            * self.y_padding
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

        if (
            x2 <= x1
            or y2 <= y1
        ):

            return None

        mouth_crop = image[
            y1:y2,
            x1:x2
        ]

        if mouth_crop.size == 0:

            return None

        return {
            "crop": mouth_crop,
            "box": (
                x1,
                y1,
                x2,
                y2
            ),
            "points": mouth_points,
        }


    # ========================================================
    # PROCESS ONE FRAME
    # ========================================================

    def process_frame(
        self,
        frame
    ):

        if frame is None:

            raise ValueError(
                "Input frame is None."
            )

        self._frame_counter += 1

        if len(frame.shape) != 3:

            raise ValueError(
                "Expected BGR color image."
            )


        height, width = (
            frame.shape[:2]
        )


        # ====================================================
        # RESULT STRUCTURE
        # ====================================================

        result = {

            "frame_width": width,

            "frame_height": height,

            "eyes": [],

            "phone": {
                "detected": False,
                "confidence": 0.0,
                "box": None,
            },

            "seatbelt": {
                "detected": False,
                "confidence": 0.0,
                "box": None,
            },

            "face": {
                "detected": False,
            },

            "mouth": {
                "state": "UNKNOWN",
                "confidence": 0.0,
                "box": None,
                "yawning_probability": 0.0,
                "non_yawning_probability": 0.0,
            },
        }


        # ====================================================
        # YOLO
        # ====================================================

        yolo_results = (
            self.yolo_model.predict(
                source=frame,
                conf=self.yolo_confidence,
                imgsz=self.yolo_image_size,
                max_det=self.yolo_max_detections,
                device="cpu",
                verbose=False
            )
        )

        if yolo_results:

            yolo_result = (
                yolo_results[0]
            )

            if (
                yolo_result.boxes
                is not None
            ):

                boxes = (
                    yolo_result.boxes
                )

                classes = (
                    boxes.cls
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidences = (
                    boxes.conf
                    .cpu()
                    .numpy()
                )

                xyxy = (
                    boxes.xyxy
                    .cpu()
                    .numpy()
                )


                # --------------------------------------------
                # Process YOLO detections
                # --------------------------------------------

                for cls_id, confidence, box in zip(
                    classes,
                    confidences,
                    xyxy
                ):

                    confidence = float(
                        confidence
                    )

                    x1, y1, x2, y2 = (
                        box.astype(int)
                    )

                    x1 = max(
                        0,
                        min(width - 1, x1)
                    )

                    y1 = max(
                        0,
                        min(height - 1, y1)
                    )

                    x2 = max(
                        0,
                        min(width, x2)
                    )

                    y2 = max(
                        0,
                        min(height, y2)
                    )


                    # ----------------------------------------
                    # Eye
                    # ----------------------------------------

                    if cls_id in {
                        self.OPEN_EYE_CLASS,
                        self.CLOSED_EYE_CLASS
                    }:

                        if (
                            x2 <= x1
                            or y2 <= y1
                        ):

                            continue

                        eye_crop = frame[
                            y1:y2,
                            x1:x2
                        ]

                        if (
                            eye_crop.size == 0
                        ):

                            continue

                        eye_result = (
                            self._predict_eye(
                                eye_crop
                            )
                        )

                        eye_result[
                            "box"
                        ] = (
                            x1,
                            y1,
                            x2,
                            y2
                        )

                        eye_result[
                            "yolo_class"
                        ] = int(cls_id)

                        eye_result[
                            "yolo_confidence"
                        ] = confidence

                        result[
                            "eyes"
                        ].append(
                            eye_result
                        )


                    # ----------------------------------------
                    # Phone
                    # ----------------------------------------

                    elif cls_id == (
                        self.PHONE_CLASS
                    ):

                        if (
                            not result[
                                "phone"
                            ]["detected"]
                            or confidence
                            > result[
                                "phone"
                            ]["confidence"]
                        ):

                            result[
                                "phone"
                            ] = {
                                "detected": True,
                                "confidence": confidence,
                                "box": (
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                ),
                            }


                    # ----------------------------------------
                    # Seatbelt
                    # ----------------------------------------

                    elif cls_id == (
                        self.SEATBELT_CLASS
                    ):

                        if (
                            not result[
                                "seatbelt"
                            ]["detected"]
                            or confidence
                            > result[
                                "seatbelt"
                            ]["confidence"]
                        ):

                            result[
                                "seatbelt"
                            ] = {
                                "detected": True,
                                "confidence": confidence,
                                "box": (
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                ),
                            }


        # ====================================================
        # MEDIAPIPE / MOUTH
        # ====================================================

        # MediaPipe + mouth CNN are the most useful part of the pipeline
        # to decimate slightly for CPU live use. Every second frame is
        # refreshed; temporal_engine supplies the persistence between
        # refreshed observations. This keeps the camera responsive without
        # changing the trained mouth model.
        use_cached_mouth = (
            self._last_mouth_result is not None
            and self._frame_counter % self.mouth_inference_interval != 0
        )

        if use_cached_mouth:
            cached = self._last_mouth_result
            result["face"]["detected"] = bool(cached.get("face_detected", False))
            result["mouth"] = dict(cached.get("mouth", result["mouth"]))
        else:
            image_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )

            try:
                detection_result = self.face_landmarker.detect(mp_image)
            except Exception:
                detection_result = None

            if (
                detection_result is not None
                and detection_result.face_landmarks
            ):
                result["face"]["detected"] = True

                landmarks = detection_result.face_landmarks[0]
                mouth_roi = self._get_mouth_roi(frame, landmarks)

                if mouth_roi is not None:
                    mouth_result = self._predict_mouth(mouth_roi["crop"])
                    result["mouth"] = {
                        "state": mouth_result["state"],
                        "confidence": mouth_result["confidence"],
                        "box": mouth_roi["box"],
                        "yawning_probability": mouth_result["yawning_probability"],
                        "non_yawning_probability": mouth_result["non_yawning_probability"],
                    }

                self._last_mouth_result = {
                    "face_detected": result["face"]["detected"],
                    "mouth": dict(result["mouth"]),
                }
            else:
                # Do not retain a stale mouth result indefinitely. The
                # temporal engine has its own short grace period.
                self._last_mouth_result = {
                    "face_detected": False,
                    "mouth": dict(result["mouth"]),
                }


        # ====================================================
        # RETURN
        # ====================================================

        return result


    # ========================================================
    # RELEASE RESOURCES
    # ========================================================

    def close(self):

        if hasattr(
            self,
            "face_landmarker"
        ):

            self.face_landmarker.close()