"""
Flask backend for the Driver Drowsiness / Driver Monitoring System.

Architecture:
    Streamlit/WebRTC frontend
            |
            | JPEG frame + session_id
            v
        Flask /api/frame
            |
            +--> DriverInferenceEngine
            +--> DriverTemporalEngine
            +--> CentralSafetyEngine
            +--> AlertManager
            |
            v
        JSON safety result

Run from the PROJECT ROOT:
    python -m backend.app
"""

from __future__ import annotations

import atexit
import base64
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------------------------------------------------------------------
# Make the repository root importable.
# backend/app.py -> backend -> project root
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inference_engine import DriverInferenceEngine
from src.inference.temporal_engine import DriverTemporalEngine
from src.inference.central_safety_engine import CentralSafetyEngine
from src.inference.alert_manager import AlertManager


app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------
# Model/inference engine
#
# The three trained models are loaded once when the Flask process starts.
# This is important: DO NOT load YOLO/CNN models for every camera frame.
# ---------------------------------------------------------------------
print("=" * 70)
print("DRIVER MONITORING BACKEND")
print("=" * 70)
print(f"Project root: {PROJECT_ROOT}")
print("Loading AI inference engine...")

inference_engine = DriverInferenceEngine(
    project_root=PROJECT_ROOT,
    device="cpu",
)

print("Inference engine loaded.")
print()

# MediaPipe/model inference may not be safe to execute concurrently.
# Serialize inference while keeping temporal state separate per session.
inference_lock = threading.Lock()


class DriverSession:
    """Stateful temporal/decision pipeline for one frontend session."""

    def __init__(self) -> None:
        self.temporal_engine = DriverTemporalEngine()
        self.central_engine = CentralSafetyEngine()
        self.alert_manager = AlertManager()

        self.lock = threading.RLock()
        self.frame_count = 0
        self.started_at = time.time()
        self.last_result: dict[str, Any] | None = None

    def process(self, frame: np.ndarray, capture_timestamp: float | None = None) -> dict[str, Any]:
        with self.lock:
            self.frame_count += 1

            # Prefer the timestamp captured when the browser frame entered
            # the frontend. This keeps temporal analysis tied to real camera
            # time even when CPU inference takes longer than one frame period.
            timestamp = (
                float(capture_timestamp)
                if capture_timestamp is not None
                else time.monotonic()
            )

            inference_start = time.perf_counter()

            with inference_lock:
                inference_result = inference_engine.process_frame(frame)

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            temporal_result = self.temporal_engine.update(
                inference_result,
                timestamp=timestamp,
            )

            safety_result = self.central_engine.update(
                inference_result,
                temporal_result,
            )

            alert_result = self.alert_manager.process(
                safety_result,
                timestamp=timestamp,
            )

            result = {
                "ok": True,
                "frame_count": self.frame_count,
                "inference_ms": round(inference_ms, 2),
                "timestamp": timestamp,
                "inference": inference_result,
                "temporal": temporal_result,
                "safety": safety_result,
                "alerts": alert_result,
            }

            self.last_result = result
            return make_json_safe(result)


sessions: dict[str, DriverSession] = {}
sessions_lock = threading.RLock()


def get_or_create_session(session_id: str) -> DriverSession:
    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = DriverSession()
        return sessions[session_id]


def delete_session(session_id: str) -> bool:
    with sessions_lock:
        return sessions.pop(session_id, None) is not None


def make_json_safe(value: Any) -> Any:
    """Convert NumPy/PyTorch scalar types into normal JSON values."""
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def decode_uploaded_frame() -> np.ndarray:
    """Decode the multipart JPEG frame sent by Streamlit."""
    uploaded = request.files.get("frame")

    if uploaded is None:
        raise ValueError("Missing multipart field: frame")

    raw = uploaded.read()
    if not raw:
        raise ValueError("Uploaded frame is empty.")

    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode uploaded image.")

    return frame


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "service": "driver-monitoring-backend",
            "project_root": str(PROJECT_ROOT),
            "model_pipeline_loaded": inference_engine is not None,
            "active_sessions": len(sessions),
        }
    )


@app.post("/api/session/start")
def start_session():
    session_id = request.json.get("session_id") if request.is_json else None

    if not session_id:
        import uuid

        session_id = str(uuid.uuid4())

    get_or_create_session(session_id)

    return jsonify(
        {
            "ok": True,
            "session_id": session_id,
            "message": "Monitoring session started.",
        }
    )


@app.post("/api/session/reset")
def reset_session():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")

    if not session_id:
        return jsonify(
            {"ok": False, "error": "session_id is required."}
        ), 400

    delete_session(session_id)
    get_or_create_session(session_id)

    return jsonify(
        {
            "ok": True,
            "session_id": session_id,
            "message": "Monitoring session reset.",
        }
    )


@app.delete("/api/session/<session_id>")
def stop_session(session_id: str):
    removed = delete_session(session_id)

    return jsonify(
        {
            "ok": True,
            "session_id": session_id,
            "removed": removed,
        }
    )


@app.post("/api/frame")
def process_frame():
    session_id = request.form.get("session_id")

    if not session_id:
        return jsonify(
            {"ok": False, "error": "session_id is required."}
        ), 400

    try:
        frame = decode_uploaded_frame()
        session = get_or_create_session(session_id)

        capture_timestamp_raw = request.form.get("capture_timestamp")
        capture_timestamp = None
        if capture_timestamp_raw:
            try:
                capture_timestamp = float(capture_timestamp_raw)
            except ValueError:
                capture_timestamp = None

        result = session.process(
            frame,
            capture_timestamp=capture_timestamp,
        )
        return jsonify(result)

    except Exception as exc:
        app.logger.exception("Frame processing failed.")
        return jsonify(
            {
                "ok": False,
                "error": str(exc),
            }
        ), 500


@app.get("/api/session/<session_id>/status")
def session_status(session_id: str):
    with sessions_lock:
        session = sessions.get(session_id)

    if session is None:
        return jsonify(
            {
                "ok": False,
                "error": "Session not found.",
            }
        ), 404

    with session.lock:
        return jsonify(
            {
                "ok": True,
                "session_id": session_id,
                "frame_count": session.frame_count,
                "started_at": session.started_at,
                "last_result": make_json_safe(session.last_result),
            }
        )


@atexit.register
def cleanup():
    try:
        inference_engine.close()
    except Exception:
        pass


if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    print("Health check: http://127.0.0.1:5000/api/health")
    print()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
    )
