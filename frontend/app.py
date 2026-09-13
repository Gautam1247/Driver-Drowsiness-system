"""
Streamlit frontend for the Driver Monitoring System.

The browser camera is captured using streamlit-webrtc.
Each video frame is sent to the local Flask backend, which runs the
existing AI pipeline. The returned detections are drawn here.

Run:
    streamlit run frontend/app.py
"""

from __future__ import annotations
import uuid
import threading
import time
from typing import Any

import cv2
import numpy as np
import requests
import streamlit as st
from av import VideoFrame
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    webrtc_streamer,
)

BACKEND_URL = "http://127.0.0.1:5000"
AI_FRAME_WIDTH = 640
AI_FRAME_HEIGHT = 360


# =====================================================================
# Thread-safe result store
# =====================================================================

class LatestResultStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: dict[str, Any] | None = None
        self._error: str | None = None
        self._last_update = 0.0

    def set_result(self, result: dict[str, Any]) -> None:
        with self._lock:
            self._result = result
            self._error = None
            self._last_update = time.time()

    def set_error(self, error: str) -> None:
        with self._lock:
            self._error = error

    def get(self) -> tuple[dict[str, Any] | None, str | None, float]:
        with self._lock:
            return self._result, self._error, self._last_update


class DriverVideoProcessor(VideoProcessorBase):
    """
    WebRTC video processor with a latest-frame inference worker.

    The camera callback NEVER waits for Flask/AI inference. That is the
    critical real-time fix: the browser can continue rendering camera frames
    at its native rate while a background worker continuously processes only
    the newest available frame. Older frames are deliberately dropped rather
    than allowed to form a queue and create seconds of latency.
    """

    def __init__(self, backend_url: str, session_id: str, store: LatestResultStore):
        self.backend_url = backend_url.rstrip("/")
        self.session_id = session_id
        self.store = store

        self.http = requests.Session()
        self.frame_count = 0

        self._frame_lock = threading.Lock()
        self._pending_frame: np.ndarray | None = None
        self._pending_timestamp: float | None = None
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()

        self._latest_result: dict[str, Any] | None = None
        self._latest_result_lock = threading.Lock()

        self._worker = threading.Thread(
            target=self._inference_worker,
            name=f"DMS-Inference-{session_id[:8]}",
            daemon=True,
        )
        self._worker.start()

    def _set_latest_result(self, result: dict[str, Any]) -> None:
        with self._latest_result_lock:
            self._latest_result = result

    def _get_latest_result(self) -> dict[str, Any] | None:
        with self._latest_result_lock:
            return self._latest_result

    def _inference_worker(self) -> None:
        """Process the newest pending frame without blocking WebRTC recv()."""
        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=0.25)
            self._wake_event.clear()

            if self._stop_event.is_set():
                break

            while not self._stop_event.is_set():
                with self._frame_lock:
                    image = self._pending_frame
                    capture_timestamp = self._pending_timestamp
                    self._pending_frame = None
                    self._pending_timestamp = None

                if image is None:
                    break

                # Send a smaller AI frame to CPU inference. The displayed
                # camera remains full resolution; detection boxes are scaled
                # back to display coordinates by draw_box().
                ai_frame = cv2.resize(
                    image,
                    (AI_FRAME_WIDTH, AI_FRAME_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )

                ok, encoded = cv2.imencode(
                    ".jpg",
                    ai_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 55],
                )
                if not ok:
                    continue

                try:
                    response = self.http.post(
                        f"{self.backend_url}/api/frame",
                        files={
                            "frame": (
                                "frame.jpg",
                                encoded.tobytes(),
                                "image/jpeg",
                            )
                        },
                        data={
                            "session_id": self.session_id,
                            "capture_timestamp": str(capture_timestamp),
                        },
                        timeout=5.0,
                    )
                    response.raise_for_status()
                    result = response.json()

                    if result.get("ok"):
                        self.store.set_result(result)
                        self._set_latest_result(result)
                    else:
                        self.store.set_error(
                            result.get("error", "Backend returned an unknown error.")
                        )
                except Exception as exc:
                    self.store.set_error(str(exc))

    def recv(self, frame: VideoFrame) -> VideoFrame:
        """
        Accept a camera frame immediately.

        This method intentionally does NOT make an HTTP request. Blocking
        here was the reason the visible camera was effectively running at
        the AI inference rate (~4-5 FPS).
        """
        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        capture_timestamp = time.monotonic()

        # Latest-frame queue: replace anything waiting to be processed.
        with self._frame_lock:
            self._pending_frame = image.copy()
            self._pending_timestamp = capture_timestamp

        self._wake_event.set()

        # Return the camera frame immediately. Overlay the newest completed
        # AI result if one exists; the camera itself is never held hostage by
        # inference latency.
        annotated = image.copy()
        latest = self._get_latest_result()
        if latest is not None:
            annotated = draw_overlay(annotated, latest)

        cv2.putText(
            annotated,
            "LIVE CAMERA",
            (15, annotated.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return VideoFrame.from_ndarray(annotated, format="bgr24")

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        try:
            self.http.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


# =====================================================================
# Drawing helpers
# =====================================================================

def clamp_box(box, width: int, height: int):
    if not box or len(box) != 4:
        return None

    x1, y1, x2, y2 = [int(float(v)) for v in box]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def draw_box(
    frame: np.ndarray,
    box,
    label: str,
    thickness: int = 2,
    source_width: int | None = None,
    source_height: int | None = None,
):
    h, w = frame.shape[:2]

    # Model detections are produced on the smaller AI frame. Scale them to
    # the full-resolution WebRTC frame before drawing.
    if source_width and source_height and source_width > 0 and source_height > 0:
        sx = w / float(source_width)
        sy = h / float(source_height)
        box = [
            float(box[0]) * sx,
            float(box[1]) * sy,
            float(box[2]) * sx,
            float(box[3]) * sy,
        ] if box and len(box) == 4 else box

    coords = clamp_box(box, w, h)

    if coords is None:
        return

    x1, y1, x2, y2 = coords

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        thickness,
    )

    text_y = max(20, y1 - 8)
    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_overlay(frame: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    inference = result.get("inference", {})
    temporal = result.get("temporal", {})
    safety = result.get("safety", {})
    alerts = result.get("alerts", {})
    source_width = int(inference.get("frame_width", 0) or 0)
    source_height = int(inference.get("frame_height", 0) or 0)

    # --------------------------------------------------------------
    # Eye detections
    # --------------------------------------------------------------
    for eye in inference.get("eyes", []) or []:
        state = eye.get("state", "UNKNOWN")
        confidence = float(eye.get("confidence", 0.0))
        label = f"EYE {state} {confidence:.2f}"
        draw_box(frame, eye.get("box"), label, source_width=source_width, source_height=source_height)

    # --------------------------------------------------------------
    # Phone
    # --------------------------------------------------------------
    phone = inference.get("phone", {})
    if phone.get("detected"):
        conf = float(phone.get("confidence", 0.0))
        draw_box(
            frame,
            phone.get("box"),
            f"PHONE {conf:.2f}",
            thickness=3,
            source_width=source_width,
            source_height=source_height,
        )

    # --------------------------------------------------------------
    # Seatbelt
    # --------------------------------------------------------------
    seatbelt = inference.get("seatbelt", {})
    if seatbelt.get("detected"):
        conf = float(seatbelt.get("confidence", 0.0))
        draw_box(
            frame,
            seatbelt.get("box"),
            f"SEATBELT {conf:.2f}",
            thickness=3,
            source_width=source_width,
            source_height=source_height,
        )

    # --------------------------------------------------------------
    # Mouth
    # --------------------------------------------------------------
    mouth = inference.get("mouth", {})
    mouth_state = mouth.get("state", "NON_YAWNING")

    if mouth_state == "UNKNOWN":
        mouth_state = "NON_YAWNING"

    if mouth.get("box") is not None:
        conf = float(mouth.get("confidence", 0.0))
        draw_box(
            frame,
            mouth.get("box"),
            f"MOUTH {mouth_state} {conf:.2f}",
            source_width=source_width,
            source_height=source_height,
        )

    # --------------------------------------------------------------
    # Main safety information
    # --------------------------------------------------------------
    driver_status = safety.get("driver_status", "NORMAL")

    risk = safety.get("risk", {})
    risk_level = risk.get("level", "NORMAL")

    drowsiness = temporal.get("drowsiness", {})
    yawning = temporal.get("yawning", {})
    phone_temporal = temporal.get("phone", {})
    seatbelt_temporal = temporal.get("seatbelt", {})

    lines = [
        f"DRIVER: {driver_status}",
        f"RISK: {risk_level}",
        f"DROWSINESS: {drowsiness.get('state', 'NORMAL')}",
        f"EYE CLOSURE: {float(drowsiness.get('eye_closed_duration', 0.0)):.1f}s",
        f"PERCLOS: {float(drowsiness.get('perclos', 0.0)):.2f}",
        f"BLINKS: {drowsiness.get('blink_count', 0)}",
        f"YAWNING: {yawning.get('state', 'NON_YAWNING')}",
        f"PHONE: {phone_temporal.get('state', 'NOT_DETECTED')}",
        f"SEATBELT: {seatbelt_temporal.get('state', 'DETECTED')}",
    ]

    x = 15
    y = 25

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 23

    # --------------------------------------------------------------
    # Temporary alert notification
    # --------------------------------------------------------------
    display_alerts = alerts.get("display_alerts", []) or []

    if display_alerts:
        alert = display_alerts[0]
        message = alert.get(
            "message",
            "DRIVER SAFETY ALERT",
        )

        h, w = frame.shape[:2]

        cv2.rectangle(
            frame,
            (0, h - 90),
            (w, h),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            "!!! SAFETY ALERT !!!",
            (20, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            message[:90],
            (20, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


# =====================================================================
# Backend communication
# =====================================================================

def backend_health() -> tuple[bool, str]:
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/health",
            timeout=2.0,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("ok"):
            return True, "Backend connected."
        return False, data.get("error", "Backend is not ready.")

    except Exception as exc:
        return False, str(exc)


def start_backend_session(session_id: str) -> tuple[bool, str]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/session/start",
            json={"session_id": session_id},
            timeout=3.0,
        )
        response.raise_for_status()

        data = response.json()

        if data.get("ok"):
            return True, data.get("message", "Session started.")

        return False, data.get("error", "Could not start session.")

    except Exception as exc:
        return False, str(exc)


def reset_backend_session(session_id: str) -> tuple[bool, str]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/session/reset",
            json={"session_id": session_id},
            timeout=3.0,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("ok"):
            return True, "Monitoring session reset."

        return False, data.get("error", "Could not reset session.")

    except Exception as exc:
        return False, str(exc)


# =====================================================================

# =====================================================================
# Streamlit UI
# =====================================================================

st.set_page_config(
    page_title="DriverGuard | AI Driver Monitoring",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Bright, professional visual system
# UI ONLY: the inference/API/temporal pipeline above is unchanged.
# ---------------------------------------------------------------------

st.markdown(
    """
<style>
/* ================================================================
   GLOBAL
   ================================================================ */
.stApp {
    background: #f6f8fc;
    color: #172033;
}

.main .block-container {
    max-width: 1500px;
    padding: 1.7rem 2.1rem 2.5rem;
}

[data-testid="stHeader"] {
    background: rgba(246, 248, 252, 0.96);
}

h1, h2, h3, h4, p, label, .stMarkdown {
    color: #172033;
}

/* ================================================================
   SIDEBAR
   ================================================================ */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5eaf2;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}

section[data-testid="stSidebar"] .stButton > button {
    min-height: 42px;
    border-radius: 10px;
    font-weight: 700;
}

.sidebar-title {
    font-size: 1.18rem;
    font-weight: 800;
    color: #14213d;
    margin-bottom: 0.15rem;
}

.sidebar-subtitle {
    color: #667085;
    font-size: 0.78rem;
    margin-bottom: 1.2rem;
}

/* ================================================================
   HEADER
   ================================================================ */
.brand-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.2rem;
}

.brand-mark {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: #2563eb;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    box-shadow: 0 5px 16px rgba(37, 99, 235, 0.20);
}

.brand-name {
    font-size: 2rem;
    line-height: 1;
    font-weight: 850;
    letter-spacing: -0.04em;
    color: #14213d;
}

.brand-tagline {
    margin-top: 0.45rem;
    color: #667085;
    font-size: 0.9rem;
}

.eyebrow {
    display: inline-block;
    margin-bottom: 0.55rem;
    color: #2563eb;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}

/* ================================================================
   SECTION HEADINGS
   ================================================================ */
.section-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin: 0.1rem 0 0.65rem;
}

.section-title {
    font-size: 1rem;
    font-weight: 800;
    color: #172033;
}

.section-subtitle {
    color: #98a2b3;
    font-size: 0.73rem;
}

/* ================================================================
   CARDS
   ================================================================ */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #e3e8f0 !important;
    border-radius: 16px !important;
    background: #ffffff !important;
    box-shadow: 0 6px 24px rgba(16, 24, 40, 0.055);
}

.camera-shell {
    padding: 0.1rem 0;
}

.camera-status {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 0.8rem;
    margin-bottom: 0.7rem;
    border: 1px solid #e7ebf2;
    border-radius: 10px;
    background: #fbfcfe;
    font-size: 0.78rem;
    color: #475467;
}

.camera-status-left {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 650;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #12b76a;
    box-shadow: 0 0 0 4px #dcfae6;
}

.live-badge {
    border-radius: 999px;
    padding: 4px 9px;
    background: #ecfdf3;
    color: #027a48;
    font-size: 0.64rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}

/* ================================================================
   RISK BANNER
   ================================================================ */
.risk-banner {
    border: 1px solid #d0d5dd;
    border-radius: 13px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.9rem;
    background: #f8fafc;
}

.risk-caption {
    color: #667085;
    font-size: 0.64rem;
    font-weight: 800;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.risk-value {
    margin-top: 0.18rem;
    color: #172033;
    font-size: 1.5rem;
    line-height: 1.1;
    font-weight: 850;
}

.risk-low {
    background: #effcf4;
    border-color: #a6f4c5;
}

.risk-medium {
    background: #fffaeb;
    border-color: #fedf89;
}

.risk-high {
    background: #fff6ed;
    border-color: #fdba74;
}

.risk-critical {
    background: #fff1f0;
    border-color: #fda29b;
}

/* ================================================================
   STREAMLIT METRICS
   ================================================================ */
div[data-testid="stMetric"] {
    min-height: 88px;
    padding: 0.72rem 0.8rem;
    background: #ffffff;
    border: 1px solid #e6eaf0;
    border-radius: 12px;
    box-shadow: none;
}

div[data-testid="stMetricLabel"] p {
    color: #667085 !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.045em;
}

div[data-testid="stMetricValue"] {
    color: #172033 !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
}

.metric-section {
    color: #344054;
    font-size: 0.82rem;
    font-weight: 800;
    margin: 0.95rem 0 0.45rem;
}

/* ================================================================
   ALERT / VIOLATION AREAS
   ================================================================ */
.alert-clear {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 0.72rem 0.8rem;
    border: 1px solid #abefc6;
    border-radius: 10px;
    background: #ecfdf3;
    color: #027a48;
    font-size: 0.78rem;
    font-weight: 650;
}

.alert-clear-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #12b76a;
}

.violation-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 0.55rem;
}

.violation-chip {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 5px 9px;
    background: #fff4ed;
    color: #b93815;
    border: 1px solid #fed7aa;
    font-size: 0.68rem;
    font-weight: 750;
}

.empty-state {
    padding: 0.7rem 0.8rem;
    border: 1px solid #e4e7ec;
    border-radius: 10px;
    background: #f9fafb;
    color: #667085;
    font-size: 0.75rem;
}

/* ================================================================
   FOOTER
   ================================================================ */
.footer {
    text-align: center;
    color: #98a2b3;
    font-size: 0.68rem;
    line-height: 1.6;
    padding: 1.2rem 0 0.2rem;
}

.small-note {
    color: #98a2b3;
    font-size: 0.68rem;
}

/* ================================================================
   BUTTONS
   ================================================================ */
.stButton > button {
    border-radius: 10px;
    font-weight: 700;
}

/* WebRTC component itself remains technically unchanged. */
iframe {
    border-radius: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "store" not in st.session_state:
    st.session_state.store = LatestResultStore()

if "session_started" not in st.session_state:
    st.session_state.session_started = False

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.markdown(
    """
<div class="eyebrow">AI DRIVER SAFETY PLATFORM</div>
<div class="brand-row">
    <div class="brand-mark">DG</div>
    <div class="brand-name">DriverGuard</div>
</div>
<div class="brand-tagline">
    Real-time driver monitoring powered by YOLO, CNN, MediaPipe and temporal safety analysis.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
<div class="sidebar-title">DriverGuard</div>
<div class="sidebar-subtitle">Real-time safety console</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("#### System controls")

    if st.button("Check backend", use_container_width=True):
        ok, message = backend_health()
        if ok:
            st.success(message)
        else:
            st.error(message)

    if st.button("Reset monitoring", use_container_width=True):
        ok, message = reset_backend_session(st.session_state.session_id)
        if ok:
            st.session_state.store = LatestResultStore()
            st.success(message)
        else:
            st.error(message)

    st.markdown("---")

    st.markdown("**Backend endpoint**")
    st.code(BACKEND_URL, language="text")

    st.markdown("**Session ID**")
    st.code(st.session_state.session_id, language="text")

    st.markdown("---")
    st.markdown(
        '<div class="small-note">'
        "Local development mode · Flask and Streamlit run on this laptop."
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# Start backend session before WebRTC.
# ---------------------------------------------------------------------

if not st.session_state.session_started:
    ok, message = start_backend_session(st.session_state.session_id)

    if ok:
        st.session_state.session_started = True
    else:
        st.error(
            "Could not connect to the Flask backend.\n\n"
            f"{message}\n\n"
            "Start it first with `python -m backend.app`."
        )

# ---------------------------------------------------------------------
# Main workspace
# ---------------------------------------------------------------------

video_col, dashboard_col = st.columns([2.05, 1], gap="large")

with video_col:
    st.markdown(
        """
<div class="section-heading">
    <div class="section-title">Live camera</div>
    <div class="section-subtitle">Real-time AI video analysis</div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            """
<div class="camera-status">
    <div class="camera-status-left">
        <span class="status-dot"></span>
        Camera monitoring ready
    </div>
    <div class="live-badge">LIVE MONITOR</div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Local copies are deliberately passed to the WebRTC worker.
        # Do not access st.session_state from inside the worker factory.
        session_id = st.session_state.session_id
        store = st.session_state.store

        ctx = webrtc_streamer(
            key="driver-monitoring-camera",
            video_processor_factory=lambda: DriverVideoProcessor(
                BACKEND_URL,
                session_id,
                store,
            ),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "frameRate": {"ideal": 15, "max": 30},
                },
                "audio": False,
            },
            rtc_configuration=RTCConfiguration(
                {
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]}
                    ]
                }
            ),
        )

        if ctx.state.playing:
            st.success("Camera connected — AI monitoring is running.")
        else:
            st.info(
                "Click START above the camera preview to begin real-time monitoring."
            )

with dashboard_col:
    st.markdown(
        """
<div class="section-heading">
    <div class="section-title">Safety overview</div>
    <div class="section-subtitle">Live status</div>
</div>
""",
        unsafe_allow_html=True,
    )

    dashboard_placeholder = st.empty()


@st.fragment(run_every="200ms")
def live_dashboard():
    result, error, last_update = st.session_state.store.get()

    if error and result is None:
        dashboard_placeholder.error(f"Backend/frame error: {error}")
        return

    if result is None:
        with dashboard_placeholder.container(border=True):
            st.markdown(
                """
<div class="risk-banner risk-low">
    <div class="risk-caption">Monitoring status</div>
    <div class="risk-value">Waiting for camera</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.info(
                "Start the camera to begin receiving live AI safety results."
            )
        return

    safety = result.get("safety", {})
    temporal = result.get("temporal", {})
    alerts = result.get("alerts", {})
    risk = safety.get("risk", {})

    driver_status = str(safety.get("driver_status", "NORMAL"))
    risk_level = str(risk.get("level", "NORMAL"))
    risk_score = risk.get("score", 0)

    drowsiness = temporal.get("drowsiness", {})
    yawning = temporal.get("yawning", {})
    phone = temporal.get("phone", {})
    seatbelt = temporal.get("seatbelt", {})

    drowsy_state = str(drowsiness.get("state", "NORMAL"))
    yawning_state = str(yawning.get("state", "NON_YAWNING"))
    phone_state = str(phone.get("state", "NOT_DETECTED"))
    seatbelt_state = str(seatbelt.get("state", "DETECTED"))

    # The dashboard intentionally exposes deterministic user-facing states.
    if drowsy_state not in {"NORMAL", "DROWSY"}:
        drowsy_state = "DROWSY" if drowsy_state == "DROWSY" else "NORMAL"
    if yawning_state not in {"NON_YAWNING", "YAWNING"}:
        yawning_state = "YAWNING" if yawning_state == "YAWNING" else "NON_YAWNING"

    active_violations = safety.get("active_violations", []) or []
    display_alerts = alerts.get("display_alerts", []) or []

    risk_key = risk_level.lower().replace(" ", "_")
    if risk_key in {"critical", "very_high"}:
        risk_class = "risk-critical"
    elif risk_key == "high":
        risk_class = "risk-high"
    elif risk_key in {"medium", "moderate"}:
        risk_class = "risk-medium"
    else:
        risk_class = "risk-low"

    with dashboard_placeholder.container(border=True):
        st.markdown(
            f"""
<div class="risk-banner {risk_class}">
    <div class="risk-caption">Current risk · score {risk_score}</div>
    <div class="risk-value">{risk_level}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Primary safety state
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Driver status", driver_status)
        with m2:
            st.metric("Drowsiness", drowsy_state)

        m3, m4 = st.columns(2)
        with m3:
            st.metric("Yawning", yawning_state)
        with m4:
            st.metric("Phone", phone_state)

        m5, m6 = st.columns(2)
        with m5:
            st.metric("Seatbelt", seatbelt_state)
        with m6:
            st.metric(
                "Eye closure",
                f'{float(drowsiness.get("eye_closed_duration", 0.0)):.2f}s',
            )

        st.markdown('<div class="metric-section">Eye analytics</div>',
                    unsafe_allow_html=True)

        e1, e2 = st.columns(2)
        with e1:
            st.metric(
                "PERCLOS",
                f'{float(drowsiness.get("perclos", 0.0)):.2f}',
            )
        with e2:
            st.metric(
                "Blink count",
                str(drowsiness.get("blink_count", 0)),
            )

        e3, e4 = st.columns(2)
        with e3:
            st.metric(
                "Avg closure",
                f'{float(drowsiness.get("average_closure_duration", 0.0)):.2f}s',
            )
        with e4:
            st.metric(
                "Inference",
                f'{float(result.get("inference_ms", 0.0)):.1f}ms',
            )

        st.markdown('<div class="metric-section">Alerts</div>',
                    unsafe_allow_html=True)

        if display_alerts:
            for alert in display_alerts:
                st.error(
                    f'{alert.get("type", "ALERT")}: '
                    f'{alert.get("message", "Safety alert detected.")}'
                )
        else:
            st.markdown(
                """
<div class="alert-clear">
    <span class="alert-clear-dot"></span>
    No active safety notification
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="metric-section">Active violations</div>',
                    unsafe_allow_html=True)

        if active_violations:
            chips = "".join(
                f'<span class="violation-chip">{str(v).replace("_", " ")}</span>'
                for v in active_violations
            )
            st.markdown(
                f'<div class="violation-wrap">{chips}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="empty-state">No confirmed safety violations.</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="small-note" style="margin-top:12px;">'
            f'Frames processed: {result.get("frame_count", 0)}'
            f' · Last update: {time.strftime("%H:%M:%S")}'
            f'</div>',
            unsafe_allow_html=True,
        )


live_dashboard()

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
<div class="footer">
    <strong>DriverGuard</strong> · AI monitoring console ·
    YOLO + Eye CNN + Mouth CNN + MediaPipe + Temporal Analysis
    <br>
    Development note: phone detection may be less reliable on some
    target-domain videos; the existing model and thresholds are unchanged.
</div>
""",
    unsafe_allow_html=True,
)
