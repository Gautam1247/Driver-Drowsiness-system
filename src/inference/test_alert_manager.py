from alert_manager import AlertManager


def make_safety_result(alerts):
    """
    Create a minimal fake Central Safety Engine result.
    """

    return {
        "driver_status": "UNSAFE",
        "risk": {
            "level": "HIGH",
            "score": 4,
        },
        "active_violations": [
            "DROWSINESS"
        ],
        "alerts": alerts,
    }


def print_result(name, result):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print("Alert Active      :", result["alert_active"])
    print("Triggered Count   :", result["triggered_count"])
    print("Suppressed Count  :", result["suppressed_count"])
    print("Total History     :", result["total_history"])

    if result["triggered_alerts"]:
        print("\nTriggered:")
        for alert in result["triggered_alerts"]:
            print(
                f"  - {alert['type']} "
                f"[{alert['severity']}]"
            )

    if result["suppressed_alerts"]:
        print("\nSuppressed:")
        for alert in result["suppressed_alerts"]:
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

    engine = AlertManager(
        cooldowns={
            "DROWSINESS_ALERT": 3.0,
            "PHONE_ALERT": 3.0,
            "SEATBELT_ALERT": 5.0,
            "YAWNING_ALERT": 5.0,
            "DROWSINESS_WARNING": 3.0,
        }
    )

    # ==============================================================
    # TEST 1 — FIRST DROWSINESS ALERT
    # ==============================================================

    drowsiness_alert = {
        "type": "DROWSINESS_ALERT",
        "severity": "CRITICAL",
        "message": "Driver appears drowsy."
    }

    result = engine.process(
        make_safety_result([drowsiness_alert]),
        timestamp=0.0
    )

    print_result(
        "TEST 1 — FIRST DROWSINESS ALERT",
        result
    )

    assert_equal(
        result["triggered_count"],
        1,
        "First alert should trigger."
    )

    assert_equal(
        result["suppressed_count"],
        0,
        "First alert should not be suppressed."
    )

    # ==============================================================
    # TEST 2 — SAME ALERT DURING COOLDOWN
    # ==============================================================

    result = engine.process(
        make_safety_result([drowsiness_alert]),
        timestamp=1.0
    )

    print_result(
        "TEST 2 — DROWSINESS DURING COOLDOWN",
        result
    )

    assert_equal(
        result["triggered_count"],
        0,
        "Alert should be suppressed during cooldown."
    )

    assert_equal(
        result["suppressed_count"],
        1,
        "Alert should be suppressed."
    )

    # ==============================================================
    # TEST 3 — AFTER COOLDOWN
    # ==============================================================

    result = engine.process(
        make_safety_result([drowsiness_alert]),
        timestamp=3.0
    )

    print_result(
        "TEST 3 — DROWSINESS AFTER COOLDOWN",
        result
    )

    assert_equal(
        result["triggered_count"],
        1,
        "Alert should trigger after cooldown."
    )

    # ==============================================================
    # TEST 4 — DIFFERENT ALERT TYPES
    # ==============================================================

    phone_alert = {
        "type": "PHONE_ALERT",
        "severity": "HIGH",
        "message": "Phone usage detected."
    }

    result = engine.process(
        make_safety_result([
            drowsiness_alert,
            phone_alert
        ]),
        timestamp=4.0
    )

    print_result(
        "TEST 4 — MULTIPLE ALERT TYPES",
        result
    )

    # Drowsiness was triggered at t=3,
    # therefore it is still in cooldown.
    #
    # Phone has never triggered, so it should trigger.

    assert_equal(
        result["triggered_count"],
        1,
        "Only the new phone alert should trigger."
    )

    assert_equal(
        result["suppressed_count"],
        1,
        "Drowsiness should still be suppressed."
    )

    # ==============================================================
    # TEST 5 — SEATBELT ALERT
    # ==============================================================

    seatbelt_alert = {
        "type": "SEATBELT_ALERT",
        "severity": "HIGH",
        "message": "Please fasten your seatbelt."
    }

    result = engine.process(
        make_safety_result([seatbelt_alert]),
        timestamp=5.0
    )

    print_result(
        "TEST 5 — SEATBELT ALERT",
        result
    )

    assert_equal(
        result["triggered_count"],
        1,
        "Seatbelt alert should trigger."
    )

    # ==============================================================
    # TEST 6 — YAWNING ALERT
    # ==============================================================

    yawning_alert = {
        "type": "YAWNING_ALERT",
        "severity": "LOW",
        "message": "Repeated yawning detected."
    }

    result = engine.process(
        make_safety_result([yawning_alert]),
        timestamp=6.0
    )

    print_result(
        "TEST 6 — YAWNING ALERT",
        result
    )

    assert_equal(
        result["triggered_count"],
        1,
        "Yawning alert should trigger."
    )

    # ==============================================================
    # TEST 7 — RESET
    # ==============================================================

    engine.reset()

    result = engine.process(
        make_safety_result([drowsiness_alert]),
        timestamp=0.0
    )

    print_result(
        "TEST 7 — RESET",
        result
    )

    assert_equal(
        result["triggered_count"],
        1,
        "Alert should trigger again after reset."
    )

    assert_equal(
        result["total_history"],
        1,
        "History should be reset."
    )

    # ==============================================================
    # TEST 8 — NO ALERTS
    # ==============================================================

    engine.reset()

    result = engine.process(
        make_safety_result([]),
        timestamp=10.0
    )

    print_result(
        "TEST 8 — NO ALERTS",
        result
    )

    assert_equal(
        result["triggered_count"],
        0,
        "No alerts should trigger."
    )

    assert_equal(
        result["suppressed_count"],
        0,
        "No alerts should be suppressed."
    )

    # ==============================================================
    # TEST 9 — ALERT HISTORY
    # ==============================================================

    engine.reset()

    engine.process(
        make_safety_result([phone_alert]),
        timestamp=0.0
    )

    engine.process(
        make_safety_result([seatbelt_alert]),
        timestamp=1.0
    )

    history = engine.get_history()

    print()
    print("=" * 70)
    print("TEST 9 — ALERT HISTORY")
    print("=" * 70)

    print("History entries:", len(history))

    for item in history:
        print(
            f"  t={item['timestamp']:.1f}s | "
            f"{item['type']} | "
            f"{item['severity']}"
        )

    assert_equal(
        len(history),
        2,
        "Two alerts should be stored in history."
    )

    # ==============================================================
    # ALL TESTS PASSED
    # ==============================================================

    print()
    print("=" * 70)
    print("ALL ALERT MANAGER TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()