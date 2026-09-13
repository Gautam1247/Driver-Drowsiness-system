from central_safety_engine import CentralSafetyEngine


def make_inference_result():
    """
    Minimal fake inference result.

    The central engine currently uses the temporal result
    for decisions, but we keep the inference structure here
    so the test resembles the real pipeline.
    """

    return {
        "frame_width": 640,
        "frame_height": 480,

        "eyes": [],

        "phone": {
            "detected": False,
            "confidence": 0.0,
            "box": None,
        },

        "seatbelt": {
            "detected": True,
            "confidence": 0.90,
            "box": None,
        },

        "face": {
            "detected": True,
        },

        "mouth": {
            "state": "NON_YAWNING",
            "confidence": 0.95,
            "box": None,
            "yawning_probability": 0.05,
            "non_yawning_probability": 0.95,
        },
    }


def make_temporal_result(
    eye_state="NORMAL",
    yawning_state="NOT_YAWNING",
    phone_state="NOT_DETECTED",
    seatbelt_state="NORMAL",
    closure_duration=0.0,
    perclos=0.0,
    blink_count=0,
    avg_closure=0.0,
    yawning_duration=0.0,
    yawning_confidence=0.0,
    phone_duration=0.0,
    phone_confidence=0.0,
    seatbelt_duration=0.0,
    seatbelt_confidence=0.0,
):
    return {
        "drowsiness": {
            "state": eye_state,
            "eye_closed_duration": closure_duration,
            "perclos": perclos,
            "blink_count": blink_count,
            "average_closure_duration": avg_closure,
            "alert": eye_state == "DROWSY",
        },

        "yawning": {
            "state": yawning_state,
            "duration": yawning_duration,
            "confidence": yawning_confidence,
        },

        "phone": {
            "state": phone_state,
            "duration": phone_duration,
            "confidence": phone_confidence,
        },

        "seatbelt": {
            "state": seatbelt_state,
            "duration": seatbelt_duration,
            "confidence": seatbelt_confidence,
        },
    }


def print_result(name, result):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print("Driver Status :", result["driver_status"])
    print("Risk Level    :", result["risk"]["level"])
    print("Risk Score    :", result["risk"]["score"])

    print("Active Violations:")
    if result["active_violations"]:
        for violation in result["active_violations"]:
            print("  -", violation)
    else:
        print("  None")

    print("Primary Violation:", result["primary_violation"])

    print("Warnings:")
    if result["warnings"]:
        for warning in result["warnings"]:
            print("  -", warning)
    else:
        print("  None")

    print("Alerts:", len(result["alerts"]))

    for alert in result["alerts"]:
        print(
            f"  - {alert['type']} "
            f"[{alert['severity']}]"
        )


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


def run_tests():

    engine = CentralSafetyEngine()

    inference = make_inference_result()

    # ==============================================================
    # TEST 1 — NORMAL DRIVER
    # ==============================================================

    temporal = make_temporal_result()

    result = engine.update(inference, temporal)

    print_result("TEST 1 — NORMAL DRIVER", result)

    assert_equal(
        result["driver_status"],
        "NORMAL",
        "Normal driver should have NORMAL status."
    )

    assert_equal(
        result["risk"]["level"],
        "NORMAL",
        "Normal driver should have NORMAL risk."
    )

    # ==============================================================
    # TEST 2 — YAWNING
    # ==============================================================

    temporal = make_temporal_result(
        yawning_state="YAWNING",
        yawning_duration=1.8,
        yawning_confidence=0.91,
    )

    result = engine.update(inference, temporal)

    print_result("TEST 2 — YAWNING", result)

    assert_equal(
        result["driver_status"],
        "CAUTION",
        "Yawning should produce CAUTION status."
    )

    assert_equal(
        result["risk"]["level"],
        "LOW",
        "Yawning should produce LOW risk."
    )

    assert "YAWNING" in result["active_violations"]

    # ==============================================================
    # TEST 3 — SEATBELT VIOLATION
    # ==============================================================

    temporal = make_temporal_result(
        seatbelt_state="VIOLATION",
        seatbelt_duration=1.5,
        seatbelt_confidence=0.92,
    )

    result = engine.update(inference, temporal)

    print_result("TEST 3 — SEATBELT VIOLATION", result)

    assert_equal(
        result["driver_status"],
        "WARNING",
        "Seatbelt violation should produce WARNING status."
    )

    assert_equal(
        result["risk"]["level"],
        "MODERATE",
        "Seatbelt violation should produce MODERATE risk."
    )

    assert "SEATBELT_VIOLATION" in result["active_violations"]

    # ==============================================================
    # TEST 4 — PHONE USAGE
    # ==============================================================

    temporal = make_temporal_result(
        phone_state="PHONE_USAGE",
        phone_duration=1.5,
        phone_confidence=0.94,
    )

    result = engine.update(inference, temporal)

    print_result("TEST 4 — PHONE USAGE", result)

    assert_equal(
        result["driver_status"],
        "UNSAFE",
        "Phone usage should produce UNSAFE status."
    )

    assert_equal(
        result["risk"]["level"],
        "HIGH",
        "Phone usage should produce HIGH risk."
    )

    assert "PHONE_USAGE" in result["active_violations"]

    # ==============================================================
    # TEST 5 — DROWSINESS
    # ==============================================================

    temporal = make_temporal_result(
        eye_state="DROWSY",
        closure_duration=2.5,
        perclos=0.42,
        blink_count=3,
        avg_closure=0.8,
    )

    result = engine.update(inference, temporal)

    print_result("TEST 5 — DROWSINESS", result)

    assert_equal(
        result["driver_status"],
        "UNSAFE",
        "Drowsiness should produce UNSAFE status."
    )

    assert_equal(
        result["risk"]["level"],
        "HIGH",
        "Drowsiness should produce HIGH risk."
    )

    assert "DROWSINESS" in result["active_violations"]

    # ==============================================================
    # TEST 6 — DROWSINESS + PHONE
    # ==============================================================

    temporal = make_temporal_result(
        eye_state="DROWSY",
        closure_duration=2.8,
        perclos=0.45,
        phone_state="PHONE_USAGE",
        phone_duration=1.5,
        phone_confidence=0.96,
    )

    result = engine.update(inference, temporal)

    print_result("TEST 6 — DROWSINESS + PHONE", result)

    assert_equal(
        result["driver_status"],
        "CRITICAL",
        "Drowsiness + phone should be CRITICAL."
    )

    assert_equal(
        result["risk"]["level"],
        "CRITICAL",
        "Drowsiness + phone should have CRITICAL risk."
    )

    assert "DROWSINESS" in result["active_violations"]
    assert "PHONE_USAGE" in result["active_violations"]

    assert_equal(
        result["primary_violation"],
        "DROWSINESS",
        "Drowsiness should have highest priority."
    )

    # ==============================================================
    # TEST 7 — DROWSINESS + YAWNING + SEATBELT
    # ==============================================================

    temporal = make_temporal_result(
        eye_state="DROWSY",
        closure_duration=3.0,
        perclos=0.50,

        yawning_state="YAWNING",
        yawning_duration=2.0,
        yawning_confidence=0.88,

        seatbelt_state="VIOLATION",
        seatbelt_duration=2.0,
        seatbelt_confidence=0.91,
    )

    result = engine.update(inference, temporal)

    print_result(
        "TEST 7 — MULTIPLE SIMULTANEOUS VIOLATIONS",
        result
    )

    assert_equal(
        result["driver_status"],
        "UNSAFE",
        "Drowsiness should dominate multiple non-phone violations."
    )

    assert_equal(
        result["risk"]["level"],
        "HIGH",
        "Drowsiness should produce HIGH risk."
    )

    assert len(result["active_violations"]) == 3

    # ==============================================================
    # TEST 8 — DROWSINESS WARNING
    # ==============================================================

    temporal = make_temporal_result(
        eye_state="WARNING",
        closure_duration=1.2,
        perclos=0.10,
    )

    result = engine.update(inference, temporal)

    print_result("TEST 8 — DROWSINESS WARNING", result)

    assert_equal(
        result["driver_status"],
        "WARNING",
        "Drowsiness warning should produce WARNING status."
    )

    assert_equal(
        result["risk"]["level"],
        "MODERATE",
        "Drowsiness warning should produce MODERATE risk."
    )

    assert "DROWSINESS_WARNING" in result["warnings"]

    # ==============================================================
    # TEST 9 — PENDING CONDITION
    # ==============================================================

    temporal = make_temporal_result(
        phone_state="PHONE_PENDING",
        phone_duration=0.5,
        phone_confidence=0.85,
    )

    result = engine.update(inference, temporal)

    print_result("TEST 9 — PHONE PENDING", result)

    assert_equal(
        result["driver_status"],
        "CAUTION",
        "Pending phone usage should produce CAUTION."
    )

    assert_equal(
        result["risk"]["level"],
        "LOW",
        "Pending phone usage should have LOW risk."
    )

    # ==============================================================
    # TEST 10 — UNKNOWN / NO RELIABLE SIGNAL
    # ==============================================================

    temporal = make_temporal_result(
        eye_state="UNKNOWN",
        yawning_state="UNKNOWN",
        phone_state="UNKNOWN",
        seatbelt_state="UNKNOWN",
    )

    result = engine.update(inference, temporal)

    print_result("TEST 10 — UNKNOWN SIGNALS", result)

    assert_equal(
        result["driver_status"],
        "NORMAL",
        "Unknown signals should not automatically create a violation."
    )

    # ==============================================================
    # ALL TESTS PASSED
    # ==============================================================

    print()
    print("=" * 70)
    print("ALL CENTRAL SAFETY ENGINE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()