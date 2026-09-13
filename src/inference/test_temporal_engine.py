from temporal_engine import DriverTemporalEngine


# ============================================================
# CREATE ENGINE
# ============================================================

engine = DriverTemporalEngine(
    eye_closure_threshold=2.0,
    warning_closure_threshold=1.0,
    yawning_threshold=1.5,
    phone_threshold=1.0,
    seatbelt_threshold=1.0,
    perclos_window=10.0,
    perclos_minimum_observation=10.0,
    perclos_warning_threshold=0.20,
    perclos_drowsy_threshold=0.30,
)


# ============================================================
# HELPER
# ============================================================

def make_result(
    eye_states,
    eye_confidence=0.99,
    mouth_state="NON_YAWNING",
    mouth_confidence=0.99,
    phone=False,
    seatbelt=True,
):

    eyes = []

    for state in eye_states:

        eyes.append({
            "state": state,
            "confidence": eye_confidence,
        })


    return {

        "eyes": eyes,

        "mouth": {
            "state": mouth_state,
            "confidence": mouth_confidence,
        },

        "phone": {
            "detected": phone,
            "confidence": 0.99,
        },

        "seatbelt": {
            "detected": seatbelt,
            "confidence": 0.99,
        },
    }


# ============================================================
# TEST 1
# NORMAL BLINK
# ============================================================

print("=" * 70)
print("TEST 1 — NORMAL BLINK")
print("=" * 70)

engine.reset()

timestamps = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
]

states = [
    ["OPEN", "OPEN"],
    ["OPEN", "OPEN"],
    ["CLOSED", "CLOSED"],
    ["CLOSED", "CLOSED"],
    ["OPEN", "OPEN"],
]

for timestamp, eyes in zip(
    timestamps,
    states
):

    result = engine.update(
        make_result(eyes),
        timestamp=timestamp
    )

    drowsiness = result[
        "drowsiness"
    ]

    print(
        f"t={timestamp:.1f}s | "
        f"Eyes={eyes} | "
        f"State={drowsiness['state']} | "
        f"Closure="
        f"{drowsiness['eye_closed_duration']:.1f}s | "
        f"Blinks="
        f"{drowsiness['blink_count']}"
    )


# ============================================================
# TEST 2
# PROLONGED CLOSURE
# ============================================================

print("\n" + "=" * 70)
print("TEST 2 — PROLONGED EYE CLOSURE")
print("=" * 70)

engine.reset()

timestamps = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
]

states = [
    ["OPEN", "OPEN"],
    ["CLOSED", "CLOSED"],
    ["CLOSED", "CLOSED"],
    ["CLOSED", "CLOSED"],
    ["CLOSED", "CLOSED"],
    ["CLOSED", "CLOSED"],
    ["OPEN", "OPEN"],
]

for timestamp, eyes in zip(
    timestamps,
    states
):

    result = engine.update(
        make_result(eyes),
        timestamp=timestamp
    )

    drowsiness = result[
        "drowsiness"
    ]

    print(
        f"t={timestamp:.1f}s | "
        f"State={drowsiness['state']} | "
        f"Closure="
        f"{drowsiness['eye_closed_duration']:.1f}s | "
        f"PERCLOS="
        f"{drowsiness['perclos']:.3f} | "
        f"Alert="
        f"{drowsiness['alert']}"
    )


# ============================================================
# TEST 3
# YAWNING
# ============================================================

print("\n" + "=" * 70)
print("TEST 3 — YAWNING")
print("=" * 70)

engine.reset()

timestamps = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
]

for timestamp in timestamps:

    result = engine.update(
        make_result(
            ["OPEN", "OPEN"],
            mouth_state="YAWNING",
            mouth_confidence=0.95,
        ),
        timestamp=timestamp
    )

    yawning = result[
        "yawning"
    ]

    print(
        f"t={timestamp:.1f}s | "
        f"State={yawning['state']} | "
        f"Duration="
        f"{yawning['duration']:.1f}s | "
        f"Alert="
        f"{yawning['alert']}"
    )


# ============================================================
# TEST 4
# LOW CONFIDENCE YAWNING
# ============================================================

print("\n" + "=" * 70)
print("TEST 4 — LOW CONFIDENCE YAWNING")
print("=" * 70)

engine.reset()

result = engine.update(
    make_result(
        ["OPEN", "OPEN"],
        mouth_state="YAWNING",
        mouth_confidence=0.59,
    ),
    timestamp=0.0
)

print(
    f"State="
    f"{result['yawning']['state']} | "
    f"Confidence=0.59"
)


# ============================================================
# TEST 5
# PHONE
# ============================================================

print("\n" + "=" * 70)
print("TEST 5 — PHONE")
print("=" * 70)

engine.reset()

timestamps = [
    0.0,
    0.5,
    1.0,
    1.5,
]

for timestamp in timestamps:

    result = engine.update(
        make_result(
            ["OPEN", "OPEN"],
            phone=True,
        ),
        timestamp=timestamp
    )

    phone = result[
        "phone"
    ]

    print(
        f"t={timestamp:.1f}s | "
        f"State={phone['state']} | "
        f"Duration="
        f"{phone['duration']:.1f}s | "
        f"Alert="
        f"{phone['alert']}"
    )


# ============================================================
# TEST 6
# SEATBELT
# ============================================================

print("\n" + "=" * 70)
print("TEST 6 — SEATBELT")
print("=" * 70)

engine.reset()

timestamps = [
    0.0,
    0.5,
    1.0,
    1.5,
]

for timestamp in timestamps:

    result = engine.update(
        make_result(
            ["OPEN", "OPEN"],
            seatbelt=False,
        ),
        timestamp=timestamp
    )

    seatbelt = result[
        "seatbelt"
    ]

    print(
        f"t={timestamp:.1f}s | "
        f"State={seatbelt['state']} | "
        f"Duration="
        f"{seatbelt['duration']:.1f}s | "
        f"Alert="
        f"{seatbelt['alert']}"
    )


# ============================================================
# TEST 7
# UNKNOWN EYES
# ============================================================

print("\n" + "=" * 70)
print("TEST 7 — UNKNOWN EYE STATE")
print("=" * 70)

engine.reset()

result = engine.update(
    make_result(
        ["OPEN"],
    ),
    timestamp=0.0
)

print(
    f"Eye state with only one detected eye: "
    f"{result['drowsiness']['state']}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL ENGINE V2 TEST COMPLETED")
print("=" * 70)