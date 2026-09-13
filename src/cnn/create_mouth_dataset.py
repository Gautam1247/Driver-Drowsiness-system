import os
import re
import random
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm.auto import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAR_CSV = PROJECT_ROOT / "evaluation" / "yawdd_mar" / "yawdd_mar_samples.csv"
SPLIT_CSV = PROJECT_ROOT / "datasets" / "mouth" / "yawdd_mirror_manifest.csv"
FACE_MODEL = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "mouth" / "cnn_dataset"
METADATA_CSV = OUTPUT_DIR / "metadata.csv"

IMAGE_SIZE = 224

# Positive threshold chosen after MAR analysis + visual inspection.
YAWNING_MAR_THRESHOLD = 0.90

# Limit highly correlated frames from one video.
MAX_POSITIVE_PER_VIDEO = 15
MAX_NEGATIVE_PER_VIDEO = 10

# High-MAR talking frames are deliberately retained as hard negatives.
HARD_TALKING_MAR_THRESHOLD = 0.90
MAX_HARD_TALKING_PER_VIDEO = 5

RANDOM_SEED = 42

# Padding around the mouth bounding box.
MOUTH_PADDING_X = 0.30
MOUTH_PADDING_Y = 0.45


# MediaPipe mouth landmark indices.
# Includes outer and inner lip landmarks.
MOUTH_LANDMARKS = [
    # Outer lips
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291,
    409, 270, 269, 267, 0,
    37, 39, 40, 185,

    # Inner lips
    78, 191, 80, 81, 82,
    13, 312, 311, 310, 415,
    308, 324, 318, 402, 317,
    14, 87, 178, 88, 95
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def clean_output_directory():
    """
    Delete the previous generated mouth CNN dataset.

    This prevents old images/metadata from mixing with the
    new dataset.
    """
    if OUTPUT_DIR.exists():
        print(f"Removing previous dataset: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        for class_name in ["non_yawning", "yawning"]:
            (OUTPUT_DIR / split / class_name).mkdir(
                parents=True,
                exist_ok=True
            )


def sanitize_filename(name):
    """
    Make a filename safe for Windows/Linux.
    """
    name = str(name)
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def evenly_select_indices(df, n):
    """
    Select approximately evenly spaced rows from a dataframe.

    Useful because adjacent frames are highly correlated.
    """
    if len(df) <= n:
        return df.copy()

    indices = np.linspace(
        0,
        len(df) - 1,
        n
    ).round().astype(int)

    indices = np.unique(indices)

    return df.iloc[indices].copy()


def diverse_sample(df, n, max_per_video, seed):
    """
    Select n samples while spreading them across videos.

    At most max_per_video samples are taken from any one video.
    """
    if len(df) == 0 or n <= 0:
        return df.iloc[0:0].copy()

    rng = np.random.default_rng(seed)

    video_groups = []

    for video, group in df.groupby("video"):
        group = group.copy()
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)

        indices = indices[:max_per_video]

        if len(indices) > 0:
            video_groups.append(list(indices))

    rng.shuffle(video_groups)

    selected = []

    # Round-robin across videos.
    while video_groups and len(selected) < n:

        remaining_groups = []

        for group_indices in video_groups:

            if group_indices:
                selected.append(group_indices.pop())

            if group_indices:
                remaining_groups.append(group_indices)

            if len(selected) >= n:
                break

        video_groups = remaining_groups

    selected = selected[:n]

    return df.loc[selected].copy()


def frame_key(row):
    """
    Unique identifier for one sampled frame.
    """
    return (
        str(row["video"]),
        int(row["frame"])
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_sources():

    print("\nLoading MAR samples...")
    mar_df = pd.read_csv(MAR_CSV)

    print(f"MAR samples: {len(mar_df):,}")

    print("\nLoading frozen YawDD subject split...")
    split_df = pd.read_csv(SPLIT_CSV)

    print(f"Videos in split manifest: {len(split_df)}")

    required_mar_columns = {
        "video",
        "behavior",
        "frame",
        "mar"
    }

    missing = required_mar_columns - set(mar_df.columns)

    if missing:
        raise ValueError(
            f"MAR CSV missing required columns: {sorted(missing)}"
        )

    required_split_columns = {
        "filename",
        "split",
        "subject_id",
        "video_path"
    }

    missing = required_split_columns - set(split_df.columns)

    if missing:
        raise ValueError(
            f"Split CSV missing required columns: {sorted(missing)}"
        )

    # Merge video information.
    df = mar_df.merge(
        split_df[
            [
                "filename",
                "split",
                "subject_id",
                "video_path"
            ]
        ],
        left_on="video",
        right_on="filename",
        how="left"
    )

    # Make sure every sampled video belongs to the frozen split.
    missing_split = df["split"].isna().sum()

    if missing_split > 0:
        raise ValueError(
            f"{missing_split} MAR samples could not be matched "
            f"to the frozen YawDD split."
        )

    return df


# ============================================================
# BUILD SAMPLE SELECTION
# ============================================================

def build_selection(df):

    selected_parts = []

    print("\n" + "=" * 70)
    print("BUILDING MOUTH CNN DATASET SELECTION")
    print("=" * 70)

    for split in ["train", "val", "test"]:

        split_df = df[df["split"] == split].copy()

        # ----------------------------------------------------
        # POSITIVE: Yawning + MAR >= 0.90
        # ----------------------------------------------------

        positive_candidates = split_df[
            (split_df["behavior"] == "Yawning") &
            (split_df["mar"] >= YAWNING_MAR_THRESHOLD)
        ].copy()

        # Spread positive samples across videos.
        positive_selected_parts = []

        for video, group in positive_candidates.groupby("video"):

            group = group.sort_values("frame")

            selected = evenly_select_indices(
                group,
                MAX_POSITIVE_PER_VIDEO
            )

            positive_selected_parts.append(selected)

        if positive_selected_parts:
            positive_selected = pd.concat(
                positive_selected_parts,
                ignore_index=False
            )
        else:
            positive_selected = positive_candidates.iloc[0:0].copy()

        positive_selected = positive_selected.copy()
        positive_selected["class_name"] = "yawning"

        # ----------------------------------------------------
        # TARGET NEGATIVE COUNT
        # ----------------------------------------------------

        target_negative_count = len(positive_selected)

        # ----------------------------------------------------
        # HARD NEGATIVES
        # Talking with high MAR.
        # ----------------------------------------------------

        hard_talking = split_df[
            (split_df["behavior"] == "Talking") &
            (split_df["mar"] >= HARD_TALKING_MAR_THRESHOLD)
        ].copy()

        hard_talking_selected = diverse_sample(
            hard_talking,
            min(target_negative_count, len(hard_talking)),
            MAX_HARD_TALKING_PER_VIDEO,
            RANDOM_SEED + 100 + len(split)
        )

        hard_keys = {
            frame_key(row)
            for _, row in hard_talking_selected.iterrows()
        }

        hard_talking_selected = hard_talking_selected.copy()
        hard_talking_selected["class_name"] = "non_yawning"

        # ----------------------------------------------------
        # REMAINING NEGATIVES
        # Normal + Talking
        # ----------------------------------------------------

        negative_candidates = split_df[
            split_df["behavior"].isin(["Normal", "Talking"])
        ].copy()

        # Remove hard negatives already selected.
        negative_candidates["_frame_key"] = negative_candidates.apply(
            frame_key,
            axis=1
        )

        negative_candidates = negative_candidates[
            ~negative_candidates["_frame_key"].isin(hard_keys)
        ].drop(columns=["_frame_key"])

        remaining_count = (
            target_negative_count -
            len(hard_talking_selected)
        )

        # Roughly divide remaining samples between
        # Normal and Talking.
        normal_target = remaining_count // 2
        talking_target = remaining_count - normal_target

        normal_candidates = negative_candidates[
            negative_candidates["behavior"] == "Normal"
        ].copy()

        talking_candidates = negative_candidates[
            negative_candidates["behavior"] == "Talking"
        ].copy()

        normal_selected = diverse_sample(
            normal_candidates,
            min(normal_target, len(normal_candidates)),
            MAX_NEGATIVE_PER_VIDEO,
            RANDOM_SEED + 200 + len(split)
        )

        talking_selected = diverse_sample(
            talking_candidates,
            min(talking_target, len(talking_candidates)),
            MAX_NEGATIVE_PER_VIDEO,
            RANDOM_SEED + 300 + len(split)
        )

        normal_selected = normal_selected.copy()
        talking_selected = talking_selected.copy()

        normal_selected["class_name"] = "non_yawning"
        talking_selected["class_name"] = "non_yawning"

        negative_selected = pd.concat(
            [
                hard_talking_selected,
                normal_selected,
                talking_selected
            ],
            ignore_index=False
        )

        # ----------------------------------------------------
        # FILL ANY SHORTFALL
        # ----------------------------------------------------

        if len(negative_selected) < target_negative_count:

            current_keys = {
                frame_key(row)
                for _, row in negative_selected.iterrows()
            }

            remaining_candidates = negative_candidates.copy()

            remaining_candidates["_frame_key"] = (
                remaining_candidates.apply(
                    frame_key,
                    axis=1
                )
            )

            remaining_candidates = remaining_candidates[
                ~remaining_candidates["_frame_key"].isin(current_keys)
            ].drop(columns=["_frame_key"])

            fill_count = (
                target_negative_count -
                len(negative_selected)
            )

            fill_selected = diverse_sample(
                remaining_candidates,
                min(fill_count, len(remaining_candidates)),
                MAX_NEGATIVE_PER_VIDEO,
                RANDOM_SEED + 400 + len(split)
            )

            fill_selected = fill_selected.copy()
            fill_selected["class_name"] = "non_yawning"

            negative_selected = pd.concat(
                [
                    negative_selected,
                    fill_selected
                ],
                ignore_index=False
            )

        # If there are somehow more negatives than positives,
        # trim them deterministically.
        if len(negative_selected) > target_negative_count:

            negative_selected = negative_selected.sample(
                n=target_negative_count,
                random_state=RANDOM_SEED
            )

        selected_split = pd.concat(
            [
                positive_selected,
                negative_selected
            ],
            ignore_index=True
        )

        selected_split["split"] = split

        selected_parts.append(selected_split)

        print(f"\n{split.upper()}")
        print("-" * 50)
        print(f"Positive candidates : {len(positive_candidates):,}")
        print(f"Positive selected   : {len(positive_selected):,}")
        print(f"Hard talking        : {len(hard_talking_selected):,}")
        print(f"Negative selected   : {len(negative_selected):,}")
        print(f"Total selected      : {len(selected_split):,}")

    selected_df = pd.concat(
        selected_parts,
        ignore_index=True
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE VIDEO/FRAME PAIRS
    # --------------------------------------------------------

    before = len(selected_df)

    selected_df = selected_df.drop_duplicates(
        subset=["video", "frame"]
    ).reset_index(drop=True)

    after = len(selected_df)

    if before != after:
        print(
            f"\nRemoved {before - after} duplicate frame selections."
        )

    return selected_df


# ============================================================
# MEDIA PIPE FACE LANDMARKER
# ============================================================

def create_face_landmarker():

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker

    FaceLandmarkerOptions = (
        mp.tasks.vision.FaceLandmarkerOptions
    )

    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(FACE_MODEL)
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return FaceLandmarker.create_from_options(options)


# ============================================================
# MOUTH CROP
# ============================================================

def crop_mouth(image, face_landmarker):

    if image is None:
        return None

    height, width = image.shape[:2]

    # MediaPipe requires RGB.
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]

    points = []

    for index in MOUTH_LANDMARKS:

        if index >= len(landmarks):
            continue

        landmark = landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    if len(points) < 10:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = max(0, min(xs))
    x_max = min(width - 1, max(xs))

    y_min = max(0, min(ys))
    y_max = min(height - 1, max(ys))

    mouth_width = x_max - x_min
    mouth_height = y_max - y_min

    # Reject unusably small crops.
    if mouth_width < 15 or mouth_height < 8:
        return None

    pad_x = int(mouth_width * MOUTH_PADDING_X)
    pad_y = int(mouth_height * MOUTH_PADDING_Y)

    x1 = max(0, x_min - pad_x)
    x2 = min(width, x_max + pad_x)

    y1 = max(0, y_min - pad_y)
    y2 = min(height, y_max + pad_y)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(
        crop,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return crop


# ============================================================
# EXTRACT CROPS
# ============================================================

def extract_crops(selected_df):

    metadata_rows = []

    grouped = selected_df.groupby(
        ["split", "video"],
        sort=True
    )

    total_groups = len(grouped)

    print("\n" + "=" * 70)
    print("EXTRACTING MOUTH CROPS")
    print("=" * 70)

    print(f"Videos to process: {total_groups}")

    successful = 0
    failed_read = 0
    failed_landmarks = 0

    with create_face_landmarker() as face_landmarker:

        for (split, video), group in tqdm(
            grouped,
            total=total_groups,
            desc="Processing videos"
        ):

            group = group.sort_values("frame")

            video_path = Path(
                str(group.iloc[0]["video_path"])
            )

            if not video_path.exists():

                print(
                    f"\nWARNING: video not found:\n"
                    f"{video_path}"
                )

                failed_read += len(group)
                continue

            cap = cv2.VideoCapture(
                str(video_path)
            )

            if not cap.isOpened():

                print(
                    f"\nWARNING: could not open:\n"
                    f"{video_path}"
                )

                failed_read += len(group)
                continue

            frame_map = {
                int(row["frame"]): row
                for _, row in group.iterrows()
            }

            max_frame = max(frame_map.keys())

            current_frame = 0

            while current_frame <= max_frame:

                ret, frame = cap.read()

                if not ret:
                    break

                if current_frame in frame_map:

                    row = frame_map[current_frame]

                    crop = crop_mouth(
                        frame,
                        face_landmarker
                    )

                    if crop is None:

                        failed_landmarks += 1

                        current_frame += 1
                        continue

                    class_name = row["class_name"]

                    output_class_dir = (
                        OUTPUT_DIR /
                        split /
                        class_name
                    )

                    video_stem = sanitize_filename(
                        Path(video).stem
                    )

                    mar_value = float(row["mar"])

                    filename = (
                        f"{class_name}_"
                        f"s{int(row['subject_id']):02d}_"
                        f"{video_stem}_"
                        f"f{int(row['frame']):06d}_"
                        f"mar{mar_value:.3f}.jpg"
                    )

                    output_path = (
                        output_class_dir /
                        filename
                    )

                    success = cv2.imwrite(
                        str(output_path),
                        crop
                    )

                    if not success:
                        print(
                            f"\nWARNING: failed to save:\n"
                            f"{output_path}"
                        )

                    else:

                        metadata_rows.append(
                            {
                                "split": split,
                                "class_name": class_name,
                                "subject_id": int(
                                    row["subject_id"]
                                ),
                                "behavior": row["behavior"],
                                "video": video,
                                "frame": int(row["frame"]),
                                "mar": mar_value,
                                "image_path": str(
                                    output_path.relative_to(
                                        PROJECT_ROOT
                                    )
                                )
                            }
                        )

                        successful += 1

                current_frame += 1

            cap.release()

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    metadata_df.to_csv(
        METADATA_CSV,
        index=False
    )

    print("\nExtraction complete.")
    print(f"Successful crops : {successful:,}")
    print(f"Read failures    : {failed_read:,}")
    print(f"Landmark failures: {failed_landmarks:,}")

    return metadata_df


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(metadata_df):

    print("\n" + "=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    if len(metadata_df) == 0:
        raise RuntimeError(
            "No mouth crops were successfully generated."
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    distribution = (
        metadata_df
        .groupby(["split", "class_name"])
        .size()
        .unstack(fill_value=0)
    )

    print(distribution)

    # --------------------------------------------------------
    # Subject leakage
    # --------------------------------------------------------

    print("\nChecking subject leakage...")

    subjects_by_split = {
        split: set(
            metadata_df[
                metadata_df["split"] == split
            ]["subject_id"].unique()
        )
        for split in ["train", "val", "test"]
    }

    train_subjects = subjects_by_split["train"]
    val_subjects = subjects_by_split["val"]
    test_subjects = subjects_by_split["test"]

    train_val_overlap = train_subjects & val_subjects
    train_test_overlap = train_subjects & test_subjects
    val_test_overlap = val_subjects & test_subjects

    if (
        train_val_overlap or
        train_test_overlap or
        val_test_overlap
    ):
        raise RuntimeError(
            "SUBJECT LEAKAGE DETECTED!\n"
            f"Train/Val: {train_val_overlap}\n"
            f"Train/Test: {train_test_overlap}\n"
            f"Val/Test: {val_test_overlap}"
        )

    print("Subject leakage check: PASSED")

    # --------------------------------------------------------
    # Check image counts on disk
    # --------------------------------------------------------

    print("\nChecking generated image files...")

    missing_files = []

    for _, row in metadata_df.iterrows():

        image_path = (
            PROJECT_ROOT /
            row["image_path"]
        )

        if not image_path.exists():
            missing_files.append(
                str(image_path)
            )

    if missing_files:

        print(
            f"Missing files: {len(missing_files)}"
        )

        raise RuntimeError(
            "Some metadata entries point to missing images."
        )

    print("Image existence check: PASSED")

    # --------------------------------------------------------
    # Check class balance
    # --------------------------------------------------------

    print("\nChecking class balance...")

    for split in ["train", "val", "test"]:

        split_counts = (
            metadata_df[
                metadata_df["split"] == split
            ]["class_name"]
            .value_counts()
        )

        yawning_count = split_counts.get(
            "yawning",
            0
        )

        non_yawning_count = split_counts.get(
            "non_yawning",
            0
        )

        print(
            f"{split.upper():5s}: "
            f"non_yawning={non_yawning_count:,}, "
            f"yawning={yawning_count:,}"
        )

        if yawning_count != non_yawning_count:

            print(
                "WARNING: class counts are not exactly equal "
                "for this split."
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\nFinal dataset:")

    print(
        f"Total images: {len(metadata_df):,}"
    )

    print(
        f"Total subjects: "
        f"{metadata_df['subject_id'].nunique()}"
    )

    print(
        f"Total videos: "
        f"{metadata_df['video'].nunique()}"
    )

    print(
        f"Metadata saved to:\n"
        f"{METADATA_CSV}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    print("=" * 70)
    print("YAWDD MOUTH CNN DATASET CREATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Verify required files
    # --------------------------------------------------------

    required_files = [
        MAR_CSV,
        SPLIT_CSV,
        FACE_MODEL
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}"
            )

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nMAR CSV:")
    print(MAR_CSV)

    print("\nSplit CSV:")
    print(SPLIT_CSV)

    print("\nFace Landmarker:")
    print(FACE_MODEL)

    # --------------------------------------------------------
    # Clean output
    # --------------------------------------------------------

    clean_output_directory()

    # --------------------------------------------------------
    # Load source data
    # --------------------------------------------------------

    df = load_sources()

    # --------------------------------------------------------
    # Select samples
    # --------------------------------------------------------

    selected_df = build_selection(df)

    selection_csv = (
        OUTPUT_DIR /
        "selected_samples.csv"
    )

    selected_df.to_csv(
        selection_csv,
        index=False
    )

    print(
        f"\nSelection manifest saved to:\n"
        f"{selection_csv}"
    )

    # --------------------------------------------------------
    # Extract crops
    # --------------------------------------------------------

    metadata_df = extract_crops(
        selected_df
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(
        metadata_df
    )

    print("\n" + "=" * 70)
    print("MOUTH DATASET CREATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()