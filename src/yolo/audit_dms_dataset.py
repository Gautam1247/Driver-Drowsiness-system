
from pathlib import Path
from collections import Counter

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "yolo_dms"

# ============================================================
# Configuration
# ============================================================

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

VALID_CLASSES = {
    0: "open_eye",
    1: "closed_eye",
    2: "phone",
    3: "seatbelt",
}

# ============================================================
# Counters
# ============================================================

total_errors = 0

print("=" * 70)
print("FINAL DMS YOLO DATASET AUDIT")
print("=" * 70)

# ============================================================
# Audit each split
# ============================================================

for split in SPLITS:

    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    print("\n" + "=" * 70)
    print(f"{split.upper()} DATASET")
    print("=" * 70)

    if not image_dir.exists():
        print(f"ERROR: Missing image directory: {image_dir}")
        total_errors += 1
        continue

    if not label_dir.exists():
        print(f"ERROR: Missing label directory: {label_dir}")
        total_errors += 1
        continue

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = [
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # --------------------------------------------------------
    # Find labels
    # --------------------------------------------------------

    labels = [
        p for p in label_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".txt"
    ]

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    # --------------------------------------------------------
    # Missing labels
    # --------------------------------------------------------

    missing_labels = image_stems - label_stems

    # --------------------------------------------------------
    # Orphan labels
    # --------------------------------------------------------

    orphan_labels = label_stems - image_stems

    print(f"Images found       : {len(images)}")
    print(f"Labels found       : {len(labels)}")
    print(f"Missing labels     : {len(missing_labels)}")
    print(f"Orphan labels      : {len(orphan_labels)}")

    if missing_labels:
        print("\nFirst missing labels:")
        for name in sorted(missing_labels)[:10]:
            print(" ", name)

        total_errors += len(missing_labels)

    if orphan_labels:
        print("\nFirst orphan labels:")
        for name in sorted(orphan_labels)[:10]:
            print(" ", name)

        total_errors += len(orphan_labels)

    # --------------------------------------------------------
    # Annotation analysis
    # --------------------------------------------------------

    class_counts = Counter()

    total_boxes = 0
    empty_labels = 0
    invalid_boxes = 0
    invalid_classes = 0

    for label_path in labels:

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        # Empty label file
        if not lines:
            empty_labels += 1
            continue

        for line_number, line in enumerate(lines, start=1):

            parts = line.split()

            if len(parts) != 5:

                print(
                    f"\nERROR: Invalid annotation format:"
                    f"\n{label_path}"
                    f"\nLine: {line_number}"
                    f"\n{line}"
                )

                invalid_boxes += 1
                continue

            try:

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:

                print(
                    f"\nERROR: Non-numeric annotation:"
                    f"\n{label_path}"
                    f"\nLine: {line_number}"
                    f"\n{line}"
                )

                invalid_boxes += 1
                continue

            # ------------------------------------------------
            # Class validation
            # ------------------------------------------------

            if class_id not in VALID_CLASSES:

                print(
                    f"\nERROR: Invalid class ID {class_id}"
                    f"\nFile: {label_path}"
                )

                invalid_classes += 1
                continue

            # ------------------------------------------------
            # Bounding-box validation
            # ------------------------------------------------

            values = [
                x_center,
                y_center,
                width,
                height,
            ]

            if not all(0.0 <= v <= 1.0 for v in values):

                invalid_boxes += 1

                print(
                    f"\nERROR: Bounding box outside [0,1]"
                    f"\nFile: {label_path}"
                    f"\nLine: {line_number}"
                    f"\n{line}"
                )

                continue

            if width <= 0 or height <= 0:

                invalid_boxes += 1

                print(
                    f"\nERROR: Invalid box dimensions"
                    f"\nFile: {label_path}"
                    f"\nLine: {line_number}"
                    f"\n{line}"
                )

                continue

            class_counts[class_id] += 1
            total_boxes += 1

    # --------------------------------------------------------
    # Split summary
    # --------------------------------------------------------

    print("\nAnnotation statistics:")

    print(f"Total bounding boxes : {total_boxes}")
    print(f"Empty label files    : {empty_labels}")
    print(f"Invalid classes      : {invalid_classes}")
    print(f"Invalid boxes        : {invalid_boxes}")

    print("\nClass distribution:")

    for class_id, class_name in VALID_CLASSES.items():

        print(
            f"{class_id} {class_name:12} : "
            f"{class_counts[class_id]}"
        )

    # --------------------------------------------------------
    # Split status
    # --------------------------------------------------------

    if (
        len(missing_labels) == 0
        and len(orphan_labels) == 0
        and invalid_boxes == 0
        and invalid_classes == 0
    ):

        print("\nSTATUS: PASSED")

    else:

        print("\nSTATUS: FAILED")

        total_errors += 1


# ============================================================
# Final result
# ============================================================

print("\n" + "=" * 70)
print("FINAL AUDIT RESULT")
print("=" * 70)

if total_errors == 0:

    print("DATASET AUDIT: PASSED")
    print("The YOLO dataset is ready for training.")

else:

    print(
        f"DATASET AUDIT: FAILED"
        f"\nTotal errors: {total_errors}"
    )

print("\nDataset location:")
print(DATASET_ROOT)