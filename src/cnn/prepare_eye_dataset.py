from pathlib import Path
import shutil
from collections import Counter


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "mrlEyes_2018_01"
    / "mrlEyes_2018_01"
)

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "eye"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ============================================================
# Fixed subject-aware split
# ============================================================
#
# IMPORTANT:
# Every subject appears in exactly ONE split.
#
# These groups were selected after inspecting the
# open/closed distribution of all 37 MRL subjects.
#
# The resulting validation and test sets are approximately
# 50/50 open vs closed.
# ============================================================

TEST_SUBJECTS = {
    "s0009",
    "s0012",
    "s0016",
    "s0023",
    "s0025",
    "s0034",
}

VAL_SUBJECTS = {
    "s0004",
    "s0006",
    "s0014",
    "s0015",
    "s0021",
    "s0033",
}


# ============================================================
# Helper functions
# ============================================================

def parse_filename(filename):
    """
    Parse an MRL Eye Dataset filename.

    Example:
        s0001_00001_0_0_0_0_0_01.png

    Format:
        subject_image_gender_glasses_eye_state_reflections_lighting_sensor

    Returns:
        subject_id
        eye_state
    """

    parts = Path(filename).stem.split("_")

    if len(parts) != 8:
        raise ValueError(
            f"Unexpected filename format: {filename}"
        )

    subject_id = parts[0]

    eye_state = int(parts[4])

    if eye_state not in (0, 1):
        raise ValueError(
            f"Unexpected eye-state value in {filename}: "
            f"{eye_state}"
        )

    return subject_id, eye_state


def collect_images():
    """
    Find all images and group them by subject.
    """

    images_by_subject = {}

    for image_path in RAW_DATASET_DIR.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        subject_id, eye_state = parse_filename(
            image_path.name
        )

        if subject_id not in images_by_subject:
            images_by_subject[subject_id] = []

        images_by_subject[subject_id].append(
            (image_path, eye_state)
        )

    return images_by_subject


def validate_subject_split(
    all_subjects,
    train_subjects,
    val_subjects,
    test_subjects
):
    """
    Verify that:
    1. Every subject belongs to exactly one split.
    2. No subject appears in multiple splits.
    3. No subjects are missing.
    """

    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)

    # Check overlap
    if train_set & val_set:
        raise ValueError(
            "Subject leakage detected between train and validation."
        )

    if train_set & test_set:
        raise ValueError(
            "Subject leakage detected between train and test."
        )

    if val_set & test_set:
        raise ValueError(
            "Subject leakage detected between validation and test."
        )

    # Check that every subject is included
    combined = train_set | val_set | test_set
    all_subjects_set = set(all_subjects)

    if combined != all_subjects_set:

        missing = all_subjects_set - combined
        extra = combined - all_subjects_set

        raise ValueError(
            f"Subject split mismatch.\n"
            f"Missing subjects: {missing}\n"
            f"Unknown subjects: {extra}"
        )

    print("\nSubject split validation: PASSED")
    print("No subject appears in multiple splits.")


def create_output_directories():
    """
    Create train/val/test/open/closed directories.
    """

    for split in ["train", "val", "test"]:

        for label in ["open", "closed"]:

            directory = (
                OUTPUT_DIR
                / split
                / label
            )

            directory.mkdir(
                parents=True,
                exist_ok=True
            )


def clear_existing_dataset():
    """
    Remove the previously generated eye dataset.

    The raw MRL dataset is NOT touched.
    """

    if OUTPUT_DIR.exists():

        print("\nRemoving previous prepared dataset...")

        shutil.rmtree(OUTPUT_DIR)

    print("Previous prepared dataset removed.")


def copy_images(
    images_by_subject,
    subjects,
    split_name
):
    """
    Copy images belonging to the selected subjects.
    """

    counts = Counter()

    for subject_id in sorted(subjects):

        images = images_by_subject[subject_id]

        for image_path, eye_state in images:

            if eye_state == 0:
                label = "closed"
            else:
                label = "open"

            destination_dir = (
                OUTPUT_DIR
                / split_name
                / label
            )

            destination_path = (
                destination_dir
                / image_path.name
            )

            shutil.copy2(
                image_path,
                destination_path
            )

            counts[label] += 1

    return counts


def print_split_summary(
    split_name,
    subjects,
    counts
):
    """
    Print statistics for one dataset split.
    """

    open_count = counts["open"]
    closed_count = counts["closed"]
    total = open_count + closed_count

    open_percentage = (
        open_count / total * 100
        if total > 0
        else 0
    )

    closed_percentage = (
        closed_count / total * 100
        if total > 0
        else 0
    )

    print(f"\n{split_name.upper()}")
    print("-" * 50)

    print(f"Subjects: {len(subjects)}")
    print(f"Open:    {open_count}")
    print(f"Closed:  {closed_count}")
    print(f"Total:   {total}")

    print(
        f"Distribution: "
        f"{open_percentage:.2f}% open / "
        f"{closed_percentage:.2f}% closed"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("MRL Eye Dataset Preparation")
    print("=" * 60)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nRaw dataset:")
    print(RAW_DATASET_DIR)

    print(f"\nOutput directory:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Check raw dataset
    # --------------------------------------------------------

    if not RAW_DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"{RAW_DATASET_DIR}"
        )

    # --------------------------------------------------------
    # Collect images
    # --------------------------------------------------------

    print("\nCollecting images...")

    images_by_subject = collect_images()

    all_subjects = sorted(
        images_by_subject.keys()
    )

    total_images = sum(
        len(images)
        for images in images_by_subject.values()
    )

    print(
        f"Total subjects found: {len(all_subjects)}"
    )

    print(
        f"Total images found:   {total_images}"
    )

    # --------------------------------------------------------
    # Determine train subjects
    # --------------------------------------------------------

    test_subjects = set(TEST_SUBJECTS)
    val_subjects = set(VAL_SUBJECTS)

    train_subjects = (
        set(all_subjects)
        - test_subjects
        - val_subjects
    )

    # --------------------------------------------------------
    # Validate subject split
    # --------------------------------------------------------

    validate_subject_split(
        all_subjects,
        train_subjects,
        val_subjects,
        test_subjects
    )

    # --------------------------------------------------------
    # Print subject lists
    # --------------------------------------------------------

    print("\nTRAIN SUBJECTS:")
    print(sorted(train_subjects))

    print("\nVALIDATION SUBJECTS:")
    print(sorted(val_subjects))

    print("\nTEST SUBJECTS:")
    print(sorted(test_subjects))

    # --------------------------------------------------------
    # Remove old processed dataset
    # --------------------------------------------------------

    clear_existing_dataset()

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    print("\nCreating output directories...")

    create_output_directories()

    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    print("\nCopying training images...")

    train_counts = copy_images(
        images_by_subject,
        train_subjects,
        "train"
    )

    print("Copying validation images...")

    val_counts = copy_images(
        images_by_subject,
        val_subjects,
        "val"
    )

    print("Copying test images...")

    test_counts = copy_images(
        images_by_subject,
        test_subjects,
        "test"
    )

    # --------------------------------------------------------
    # Print summaries
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print_split_summary(
        "train",
        train_subjects,
        train_counts
    )

    print_split_summary(
        "validation",
        val_subjects,
        val_counts
    )

    print_split_summary(
        "test",
        test_subjects,
        test_counts
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    prepared_total = (
        sum(train_counts.values())
        + sum(val_counts.values())
        + sum(test_counts.values())
    )

    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)

    print(f"Original images:  {total_images}")
    print(f"Prepared images:  {prepared_total}")

    if total_images == prepared_total:
        print("Image count verification: PASSED")
    else:
        print("WARNING: Image count mismatch!")

    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    main()