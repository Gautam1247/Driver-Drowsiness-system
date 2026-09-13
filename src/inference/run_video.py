"""
Real-Time Driver Monitoring - Video Runner

Pipeline:

Video
  ↓
Inference Engine
  ↓
Temporal Engine
  ↓
Central Safety Engine
  ↓
Alert Manager
  ↓
Annotated Video

This version processes a prerecorded video file.
"""

import sys
import time
from pathlib import Path

import cv2

# ----------------------------------------------------------------------
# Make imports work when this file is executed directly
# ----------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from inference_engine import DriverInferenceEngine
from temporal_engine import DriverTemporalEngine
from central_safety_engine import CentralSafetyEngine
from alert_manager import AlertManager


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

VIDEO_PATH = PROJECT_ROOT / "videos" / "driver_test.mp4"

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "video_output"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = OUTPUT_DIR / "driver_monitoring_output.mp4"


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------

def draw_text(
    frame,
    text,
    position,
    scale=0.6,
    thickness=2,
):
    """
    Draw readable text on frame.
    """

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def get_current_eye_state(inference_result):
    """Return the current frame-level eye state for the UI."""
    eyes = inference_result.get("eyes", [])

    if len(eyes) < 2:
        return "UNKNOWN"

    states = []
    for eye in eyes:
        state = eye.get("state", "UNKNOWN")
        if state in {"OPEN", "CLOSED"}:
            states.append(state)

    if len(states) < 2:
        return "UNKNOWN"

    if all(state == "OPEN" for state in states):
        return "OPEN"

    if all(state == "CLOSED" for state in states):
        return "CLOSED"

    return "MIXED"


def draw_panel(
    frame,
    safety_result,
    temporal_result,
    inference_result,
    alert_result,
):
    """
    Draw driver safety information on the frame.
    """

    height, width = frame.shape[:2]

    # --------------------------------------------------------------
    # Top information panel
    # --------------------------------------------------------------

    panel_height = 245

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, panel_height),
        (0, 0, 0),
        -1,
    )

    # Slight transparency
    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame,
    )

    # --------------------------------------------------------------
    # Safety status
    # --------------------------------------------------------------

    driver_status = safety_result.get(
        "driver_status",
        "UNKNOWN"
    )

    risk = safety_result.get(
        "risk",
        {}
    )

    risk_level = risk.get(
        "level",
        "UNKNOWN"
    )

    risk_score = risk.get(
        "score",
        0
    )

    draw_text(
        frame,
        f"DRIVER STATUS: {driver_status}",
        (20, 35),
        scale=0.8,
        thickness=2,
    )

    draw_text(
        frame,
        f"RISK: {risk_level}   SCORE: {risk_score}",
        (20, 65),
        scale=0.65,
    )

    # --------------------------------------------------------------
    # Active violations
    # --------------------------------------------------------------

    violations = safety_result.get(
        "active_violations",
        []
    )

    if violations:
        violation_text = ", ".join(violations)
    else:
        violation_text = "NONE"

    draw_text(
        frame,
        f"VIOLATIONS: {violation_text}",
        (20, 95),
        scale=0.55,
    )

    # --------------------------------------------------------------
    # Temporal eye information
    # --------------------------------------------------------------

    drowsiness = temporal_result.get(
        "drowsiness",
        {}
    )

    # IMPORTANT:
    # Show the actual current eye state from frame-level inference.
    # drowsiness["state"] is a temporal state such as NORMAL, WARNING,
    # or DROWSY and should not be displayed as the eye state.
    eye_status = get_current_eye_state(
        inference_result
    )

    drowsiness_state = drowsiness.get(
        "state",
        "UNKNOWN"
    )

    closure_duration = drowsiness.get(
        "eye_closed_duration",
        0.0
    )

    perclos = drowsiness.get(
        "perclos",
        0.0
    )

    blink_count = drowsiness.get(
        "blink_count",
        0
    )

    average_closure = drowsiness.get(
        "average_closure_duration",
        0.0
    )

    draw_text(
        frame,
        (
            f"EYES: {eye_status}   "
            f"CLOSURE: {closure_duration:.1f}s   "
            f"PERCLOS: {perclos:.2f}"
        ),
        (20, 125),
        scale=0.52,
    )

    draw_text(
        frame,
        (
            f"BLINKS: {blink_count}   "
            f"AVG CLOSURE: {average_closure:.2f}s"
        ),
        (20, 150),
        scale=0.50,
    )

    draw_text(
        frame,
        f"DROWSINESS: {drowsiness_state}",
        (20, 175),
        scale=0.50,
    )

    # --------------------------------------------------------------
    # Yawning temporal information
    # --------------------------------------------------------------

    yawning = temporal_result.get(
        "yawning",
        {}
    )

    yawning_state = yawning.get(
        "state",
        "UNKNOWN"
    )

    yawning_duration = yawning.get(
        "duration",
        0.0
    )

    draw_text(
        frame,
        (
            f"YAWN: {yawning_state}   "
            f"DURATION: {yawning_duration:.1f}s"
        ),
        (20, 200),
        scale=0.48,
    )

    # --------------------------------------------------------------
    # Temporary safety alert indicator
    # --------------------------------------------------------------
    #
    # IMPORTANT:
    # Do not use safety_result["alert_active"] here.
    # That value represents the current safety condition and may remain
    # active for many frames.
    #
    # AlertManager instead provides a short-lived UI notification.
    # --------------------------------------------------------------

    alert_active = alert_result.get(
        "alert_active",
        False
    )

    display_alerts = alert_result.get(
        "display_alerts",
        []
    )

    if alert_active and display_alerts:

        latest_alert = display_alerts[-1]

        alert_type = latest_alert.get(
            "type",
            "SAFETY_ALERT"
        )

        draw_text(
            frame,
            f"!!! {alert_type} !!!",
            (20, 230),
            scale=0.60,
            thickness=2,
        )

    return frame


def draw_detections(
    frame,
    inference_result,
    temporal_result,
):
    """
    Draw YOLO eye/phone/seatbelt detections and mouth ROI.
    """

    # --------------------------------------------------------------
    # Eyes
    # --------------------------------------------------------------

    eyes = inference_result.get(
        "eyes",
        []
    )

    for index, eye in enumerate(eyes, start=1):

        box = eye.get("box")

        if not box:
            continue

        x1, y1, x2, y2 = map(
            int,
            box
        )

        state = eye.get(
            "state",
            "UNKNOWN"
        )

        confidence = eye.get(
            "confidence",
            0.0
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        draw_text(
            frame,
            (
                f"Eye {index}: "
                f"{state} "
                f"{confidence:.2f}"
            ),
            (x1, max(20, y1 - 8)),
            scale=0.45,
            thickness=1,
        )

    # --------------------------------------------------------------
    # Phone
    # --------------------------------------------------------------

    phone = inference_result.get(
        "phone",
        {}
    )

    if phone.get("detected", False):

        box = phone.get("box")

        if box:

            x1, y1, x2, y2 = map(
                int,
                box
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            draw_text(
                frame,
                (
                    f"PHONE "
                    f"{phone.get('confidence', 0.0):.2f}"
                ),
                (x1, max(20, y1 - 8)),
                scale=0.5,
            )

    # --------------------------------------------------------------
    # Seatbelt
    # --------------------------------------------------------------

    seatbelt_inference = inference_result.get(
        "seatbelt",
        {}
    )

    seatbelt_temporal = temporal_result.get(
        "seatbelt",
        {}
    )

    # Prefer the temporally smoothed seatbelt display state.
    # During a short YOLO dropout this keeps the last reliable
    # seatbelt box visible instead of making it disappear instantly.
    seatbelt_display_detected = seatbelt_temporal.get(
        "display_detected",
        seatbelt_inference.get("detected", False)
    )

    if seatbelt_display_detected:

        box = seatbelt_temporal.get(
            "display_box",
            seatbelt_inference.get("box")
        )

        if box:

            x1, y1, x2, y2 = map(
                int,
                box
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            draw_text(
                frame,
                (
                    f"SEATBELT "
                    f"{seatbelt_temporal.get(
                        'display_confidence',
                        seatbelt_inference.get(
                            'confidence',
                            0.0
                        )
                    ):.2f}"
                ),
                (x1, max(20, y1 - 8)),
                scale=0.5,
            )

    # --------------------------------------------------------------
    # Mouth
    # --------------------------------------------------------------

    mouth = inference_result.get(
        "mouth",
        {}
    )

    mouth_box = mouth.get("box")

    if mouth_box:

        x1, y1, x2, y2 = map(
            int,
            mouth_box
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        draw_text(
            frame,
            (
                f"MOUTH: "
                f"{mouth.get('state', 'UNKNOWN')} "
                f"{mouth.get('confidence', 0.0):.2f}"
            ),
            (x1, max(20, y1 - 8)),
            scale=0.45,
            thickness=1,
        )

    return frame


# ----------------------------------------------------------------------
# Main video runner
# ----------------------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("REAL-TIME DRIVER MONITORING")
    print("=" * 70)

    # --------------------------------------------------------------
    # Check video
    # --------------------------------------------------------------

    if not VIDEO_PATH.exists():

        print()
        print("ERROR: Video file not found.")
        print()
        print("Expected:")
        print(VIDEO_PATH)
        print()
        print(
            "Put your test video inside the "
            "'videos' folder and name it:"
        )
        print("driver_test.mp4")
        print()

        return

    print()
    print("Video:", VIDEO_PATH)

    # --------------------------------------------------------------
    # Open video
    # --------------------------------------------------------------

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        print("ERROR: Could not open video.")

        return

    # --------------------------------------------------------------
    # Video information
    # --------------------------------------------------------------

    input_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames / input_fps
        if input_fps > 0
        else 0
    )

    print()
    print("Video Information")
    print("-----------------")
    print(f"Resolution : {frame_width}x{frame_height}")
    print(f"Input FPS  : {input_fps:.2f}")
    print(f"Frames     : {total_frames}")
    print(f"Duration   : {duration:.2f}s")

    # --------------------------------------------------------------
    # Output writer
    # --------------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        input_fps if input_fps > 0 else 30.0,
        (frame_width, frame_height),
    )

    if not writer.isOpened():

        print(
            "ERROR: Could not create output video."
        )

        cap.release()

        return

    # --------------------------------------------------------------
    # Load AI pipeline
    # --------------------------------------------------------------

    print()
    print("Loading AI pipeline...")

    inference_engine = DriverInferenceEngine(
    yolo_confidence=0.05
)

    temporal_engine = DriverTemporalEngine()

    central_engine = CentralSafetyEngine()

    alert_manager = AlertManager()

    print("AI pipeline loaded successfully.")

    # --------------------------------------------------------------
    # Processing statistics
    # --------------------------------------------------------------

    frame_count = 0

    processing_start = time.perf_counter()

    total_inference_time = 0.0

    # --------------------------------------------------------------
    # Video loop
    # --------------------------------------------------------------

    print()
    print("Starting video processing...")
    print("Press Q to stop.")
    print()

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # ------------------------------------------------------
            # Use video timestamp for temporal analysis
            # ------------------------------------------------------

            if input_fps > 0:

                video_timestamp = (
                    (frame_count - 1)
                    / input_fps
                )

            else:

                video_timestamp = (
                    frame_count - 1
                ) / 30.0

            # ------------------------------------------------------
            # AI inference
            # ------------------------------------------------------

            inference_start = time.perf_counter()

            inference_result = (
                inference_engine.process_frame(
                    frame
                )
            )
            if frame_count <= 3:
                print()
                print("=" * 70)
                print(f"DEBUG FRAME {frame_count}")
                print("=" * 70)

                print("INFERENCE EYES:")
                print(inference_result.get("eyes"))

                print()
                print("FULL INFERENCE RESULT:")
                print(inference_result)

            inference_end = time.perf_counter()

            total_inference_time += (
                inference_end -
                inference_start
            )

            # ------------------------------------------------------
            # Temporal analysis
            # ------------------------------------------------------

            temporal_result = (
                temporal_engine.update(
                    inference_result,
                    timestamp=video_timestamp,
                )
            )
            if frame_count <= 3:
                print()
                print("TEMPORAL RESULT:")
                print(temporal_result)
                print("=" * 70)
            # ------------------------------------------------------
            # Central safety reasoning
            # ------------------------------------------------------

            safety_result = (
                central_engine.update(
                    inference_result,
                    temporal_result,
                )
            )

            # ------------------------------------------------------
            # Alert manager
            # ------------------------------------------------------

            alert_result = (
                alert_manager.process(
                    safety_result,
                    timestamp=video_timestamp,
                )
            )

            # ------------------------------------------------------
            # Draw detections
            # ------------------------------------------------------

            frame = draw_detections(
                frame,
                inference_result,
                temporal_result,
            )

            # ------------------------------------------------------
            # Draw safety panel
            # ------------------------------------------------------

            frame = draw_panel(
                frame,
                safety_result,
                temporal_result,
                inference_result,
                alert_result,
            )

            # ------------------------------------------------------
            # Write output
            # ------------------------------------------------------

            writer.write(frame)

            # ------------------------------------------------------
            # Display
            # ------------------------------------------------------

            cv2.imshow(
                "Driver Monitoring System",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                print()
                print(
                    "Stopped by user."
                )

                break

            # ------------------------------------------------------
            # Progress
            # ------------------------------------------------------

            if frame_count % 30 == 0:

                elapsed = (
                    time.perf_counter()
                    - processing_start
                )

                processing_fps = (
                    frame_count / elapsed
                    if elapsed > 0
                    else 0
                )

                progress = (
                    frame_count /
                    total_frames * 100
                    if total_frames > 0
                    else 0
                )

                current_eye_state = get_current_eye_state(
                    inference_result
                )

                drowsiness = temporal_result.get(
                    "drowsiness",
                    {}
                )

                driver_status = safety_result.get(
                    "driver_status",
                    "UNKNOWN"
                )

                risk = safety_result.get(
                    "risk",
                    {}
                )

                print(
                    f"[UI DEBUG] Frame {frame_count} | "
                    f"Driver={driver_status} | "
                    f"Risk={risk.get('level', 'UNKNOWN')} "
                    f"Score={risk.get('score', 0)} | "
                    f"Eyes={current_eye_state} | "
                    f"Drowsiness={drowsiness.get('state', 'UNKNOWN')} | "
                    f"Closure={drowsiness.get('eye_closed_duration', 0.0):.2f}s | "
                    f"PERCLOS={drowsiness.get('perclos', 0.0):.3f} | "
                    f"Blinks={drowsiness.get('blink_count', 0)}"
                )

                print(
                    f"Frame {frame_count}/"
                    f"{total_frames} "
                    f"({progress:.1f}%) | "
                    f"Processing FPS: "
                    f"{processing_fps:.2f}"
                )

    finally:

        # ----------------------------------------------------------
        # Cleanup
        # ----------------------------------------------------------

        cap.release()

        writer.release()

        cv2.destroyAllWindows()

        inference_engine.close()

    # --------------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------------

    total_time = (
        time.perf_counter()
        - processing_start
    )

    processing_fps = (
        frame_count / total_time
        if total_time > 0
        else 0
    )

    average_inference_ms = (
        total_inference_time /
        frame_count *
        1000
        if frame_count > 0
        else 0
    )

    history = (
        alert_manager.get_history()
    )

    print()
    print("=" * 70)
    print("VIDEO PROCESSING COMPLETED")
    print("=" * 70)

    print()
    print("Performance")
    print("-----------")
    print(f"Frames processed : {frame_count}")
    print(f"Processing time  : {total_time:.2f}s")
    print(f"Processing FPS   : {processing_fps:.2f}")
    print(
        f"Avg inference    : "
        f"{average_inference_ms:.2f} ms/frame"
    )

    print()
    print("Alerts")
    print("------")
    print(
        f"Alerts delivered: "
        f"{len(history)}"
    )

    print()
    print("Output:")
    print(OUTPUT_PATH)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()