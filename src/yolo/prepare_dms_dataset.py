from pathlib import Path
import shutil

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = PROJECT_ROOT / "datasets" / "raw" / "dms"

OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "yolo_dms"

# ============================================================
# Original DMS class mapping
# ============================================================

# Original dataset:
# 0 = Open Eye
# 1 = Closed Eye
# 2 = Cigarette
# 3 = Phone
# 4 = Seatbelt

# Our project:
# 0 = open_eye
# 1 = closed_eye
# 2 = phone
# 3 = seatbelt

CLASS_MAPPING = {
    0: 0,   # Open Eye -> open_eye
    1: 1,   # Closed Eye -> closed_eye
    3: 2,   # Phone -> phone
    4: 3,   # Seatbelt -> seatbelt
}

CLASS_NAMES = {
    0: "open_eye",
    1: "closed_eye",
    2: "phone",
    3: "seatbelt",
}

SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# ============================================================
# Validation counters
# ============================================================

stats = {
    split: {
        "images": 0,
        "labels": 0,
        "boxes": 0,
        "open_eye": 0,
        "closed_eye": 0,
        "phone": 0,
        "seatbelt": 0,
        "cigarette_removed": 0,
    }
    for split in SPLITS.values()
}

# ============================================================
# Clean previous generated dataset
# ============================================================

if OUTPUT_ROOT.exists():
    print(f"Removing previous generated dataset:")
    print(OUTPUT_ROOT)
    shutil.rmtree(OUTPUT_ROOT)

# ============================================================
# Create output directories
# ============================================================

for split in SPLITS.values():
    (OUTPUT_ROOT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_ROOT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )

# ============================================================
# Process each split
# ============================================================

for source_split, output_split in SPLITS.items():

    source_images = SOURCE_ROOT / source_split / "images"
    source_labels = SOURCE_ROOT / source_split / "labels"

    output_images = OUTPUT_ROOT / "images" / output_split
    output_labels = OUTPUT_ROOT / "labels" / output_split

    print("\n" + "=" * 60)
    print(f"Processing: {source_split} -> {output_split}")
    print("=" * 60)

    image_files = [
        p for p in source_images.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for image_path in image_files:

        label_path = source_labels / f"{image_path.stem}.txt"

        # ----------------------------------------------------
        # Make sure label exists
        # ----------------------------------------------------

        if not label_path.exists():
            print(f"WARNING: Missing label for {image_path.name}")
            continue

        # ----------------------------------------------------
        # Read annotations
        # ----------------------------------------------------

        output_annotations = []

        with open(label_path, "r", encoding="utf-8") as f:

            for line_number, line in enumerate(f, start=1):

                line = line.strip()

                # Ignore blank lines
                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(
                        f"Invalid annotation in {label_path}\n"
                        f"Line {line_number}: {line}"
                    )

                original_class = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # ------------------------------------------------
                # Validate bounding box
                # ------------------------------------------------

                values = [
                    x_center,
                    y_center,
                    width,
                    height,
                ]

                if not all(0.0 <= v <= 1.0 for v in values):
                    raise ValueError(
                        f"Bounding box outside [0,1] in "
                        f"{label_path}\n"
                        f"Line {line_number}: {line}"
                    )

                if width <= 0 or height <= 0:
                    raise ValueError(
                        f"Invalid box dimensions in "
                        f"{label_path}\n"
                        f"Line {line_number}: {line}"
                    )

                # ------------------------------------------------
                # Remove cigarette
                # ------------------------------------------------

                if original_class == 2:

                    stats[output_split][
                        "cigarette_removed"
                    ] += 1

                    continue

                # ------------------------------------------------
                # Keep our four classes
                # ------------------------------------------------

                if original_class not in CLASS_MAPPING:

                    raise ValueError(
                        f"Unexpected class {original_class} "
                        f"in {label_path}"
                    )

                new_class = CLASS_MAPPING[original_class]

                output_annotations.append(
                    f"{new_class} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

                stats[output_split]["boxes"] += 1

                if new_class == 0:
                    stats[output_split]["open_eye"] += 1

                elif new_class == 1:
                    stats[output_split]["closed_eye"] += 1

                elif new_class == 2:
                    stats[output_split]["phone"] += 1

                elif new_class == 3:
                    stats[output_split]["seatbelt"] += 1

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination_image = output_images / image_path.name

        shutil.copy2(
            image_path,
            destination_image
        )

        stats[output_split]["images"] += 1

        # ----------------------------------------------------
        # Write processed labels
        # ----------------------------------------------------

        destination_label = output_labels / label_path.name

        with open(
            destination_label,
            "w",
            encoding="utf-8"
        ) as f:

            if output_annotations:
                f.write(
                    "\n".join(output_annotations)
                    + "\n"
                )

        stats[output_split]["labels"] += 1

# ============================================================
# Create data.yaml
# ============================================================

yaml_path = OUTPUT_ROOT / "data.yaml"

yaml_content = f"""path: {OUTPUT_ROOT.as_posix()}

train: images/train
val: images/val
test: images/test

names:
  0: open_eye
  1: closed_eye
  2: phone
  3: seatbelt
"""

with open(
    yaml_path,
    "w",
    encoding="utf-8"
) as f:
    f.write(yaml_content)

# ============================================================
# Final report
# ============================================================

print("\n")
print("=" * 70)
print("DMS YOLO DATASET PREPARATION COMPLETE")
print("=" * 70)

for split in ["train", "val", "test"]:

    s = stats[split]

    print(f"\n{split.upper()}")

    print(f"Images            : {s['images']}")
    print(f"Labels            : {s['labels']}")
    print(f"Bounding boxes    : {s['boxes']}")

    print(f"Open Eye          : {s['open_eye']}")
    print(f"Closed Eye        : {s['closed_eye']}")
    print(f"Phone             : {s['phone']}")
    print(f"Seatbelt          : {s['seatbelt']}")
    print(f"Cigarettes removed: {s['cigarette_removed']}")

print("\nOutput dataset:")
print(OUTPUT_ROOT)

print("\ndata.yaml:")
print(yaml_path)

print("\nFinal classes:")

for class_id, class_name in CLASS_NAMES.items():
    print(f"{class_id} -> {class_name}")