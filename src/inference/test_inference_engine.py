from pathlib import Path

import cv2

from inference_engine import DriverInferenceEngine


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

TEST_IMAGE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "yolo_dms"
    / "images"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "unified_pipeline"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND TEST IMAGE
# ============================================================

images = sorted(
    p
    for p in TEST_IMAGE_DIR.iterdir()
    if p.suffix.lower()
    in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }
)

if not images:

    raise RuntimeError(
        "No test images found."
    )


image_path = images[0]

print("=" * 70)
print("UNIFIED INFERENCE ENGINE TEST")
print("=" * 70)

print(
    f"\nSelected image:\n{image_path}"
)


# ============================================================
# READ IMAGE
# ============================================================

frame = cv2.imread(
    str(image_path)
)

if frame is None:

    raise RuntimeError(
        "Could not read test image."
    )


# ============================================================
# LOAD ENGINE
# ============================================================

print("\nLoading inference engine...")

engine = DriverInferenceEngine(
    project_root=PROJECT_ROOT,
    device="cpu",
)

print("Inference engine ready.")


# ============================================================
# PROCESS FRAME
# ============================================================

print("\nProcessing frame...")

result = engine.process_frame(
    frame
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FRAME RESULT")
print("=" * 70)


print("\nEYES:")

if result["eyes"]:

    for index, eye in enumerate(
        result["eyes"],
        start=1
    ):

        print(
            f"  Eye {index}: "
            f"{eye['state']} "
            f"(confidence="
            f"{eye['confidence']:.3f})"
        )

else:

    print(
        "  No eyes detected."
    )


print("\nPHONE:")

print(
    f"  Detected   : "
    f"{result['phone']['detected']}"
)

print(
    f"  Confidence : "
    f"{result['phone']['confidence']:.3f}"
)


print("\nSEATBELT:")

print(
    f"  Detected   : "
    f"{result['seatbelt']['detected']}"
)

print(
    f"  Confidence : "
    f"{result['seatbelt']['confidence']:.3f}"
)


print("\nFACE:")

print(
    f"  Detected   : "
    f"{result['face']['detected']}"
)


print("\nMOUTH:")

print(
    f"  State      : "
    f"{result['mouth']['state']}"
)

print(
    f"  Confidence : "
    f"{result['mouth']['confidence']:.3f}"
)

print(
    f"  Yawning P  : "
    f"{result['mouth']['yawning_probability']:.3f}"
)


# ============================================================
# DRAW ANNOTATIONS
# ============================================================

annotated = frame.copy()


# ------------------------------------------------------------
# Eyes
# ------------------------------------------------------------

for index, eye in enumerate(
    result["eyes"],
    start=1
):

    x1, y1, x2, y2 = (
        eye["box"]
    )

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = (
        f"Eye {index}: "
        f"{eye['state']} "
        f"{eye['confidence']:.2f}"
    )

    cv2.putText(
        annotated,
        label,
        (
            x1,
            max(y1 - 8, 20)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


# ------------------------------------------------------------
# Phone
# ------------------------------------------------------------

if result["phone"]["detected"]:

    x1, y1, x2, y2 = (
        result["phone"]["box"]
    )

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated,
        f"PHONE "
        f"{result['phone']['confidence']:.2f}",
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )


# ------------------------------------------------------------
# Seatbelt
# ------------------------------------------------------------

if result["seatbelt"]["detected"]:

    x1, y1, x2, y2 = (
        result["seatbelt"]["box"]
    )

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated,
        f"SEATBELT "
        f"{result['seatbelt']['confidence']:.2f}",
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )


# ------------------------------------------------------------
# Mouth
# ------------------------------------------------------------

if (
    result["mouth"]["box"]
    is not None
):

    x1, y1, x2, y2 = (
        result["mouth"]["box"]
    )

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        (
            f"{result['mouth']['state']} "
            f"{result['mouth']['confidence']:.2f}"
        ),
        (
            x1,
            min(
                y2 + 20,
                frame.shape[0] - 10
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )


# ============================================================
# SAVE
# ============================================================

output_path = (
    OUTPUT_DIR
    / f"unified_{image_path.name}"
)

cv2.imwrite(
    str(output_path),
    annotated
)


# ============================================================
# CLEANUP
# ============================================================

engine.close()


print("\n" + "=" * 70)

print(
    "UNIFIED INFERENCE TEST COMPLETED"
)

print(
    f"\nAnnotated result:\n{output_path}"
)

print("=" * 70)