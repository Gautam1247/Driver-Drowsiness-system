from pathlib import Path
from collections import defaultdict
import random
import csv


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

YAWDD_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "YawDD.rar"
    / "YawDD"
    / "YawDD dataset"
)

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "mouth"

MANIFEST_PATH = OUTPUT_DIR / "yawdd_mirror_manifest.csv"

RANDOM_SEED = 42

TRAIN_SUBJECTS = 33
VAL_SUBJECTS = 7
TEST_SUBJECTS = 7


# ============================================================
# FIND MIRROR VIDEOS
# ============================================================

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}

videos = [
    p for p in YAWDD_ROOT.rglob("*")
    if (
        p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and "Mirror" in p.parts
    )
]

print("=" * 80)
print("                 YawDD SUBJECT SPLIT")
print("=" * 80)

print(f"\nTotal Mirror videos: {len(videos)}")


# ============================================================
# PARSE SUBJECT + BEHAVIOR
# ============================================================

subject_videos = defaultdict(list)

for video in videos:

    filename = video.stem
    parts = filename.split("-")

    if not parts[0].isdigit():
        continue

    subject_id = int(parts[0])

    filename_lower = filename.lower()

    if "-normal" in filename_lower:
        behavior = "normal"

    elif "-talking" in filename_lower:
        behavior = "talking"

    elif "-yawning" in filename_lower:
        behavior = "yawning"

    else:
        continue

    subject_videos[subject_id].append(
        {
            "path": video,
            "subject": subject_id,
            "behavior": behavior,
            "filename": video.name,
        }
    )


subjects = sorted(subject_videos.keys())

print(f"Unique subjects: {len(subjects)}")


# ============================================================
# VALIDATE SUBJECT COUNT
# ============================================================

expected_subjects = (
    TRAIN_SUBJECTS
    + VAL_SUBJECTS
    + TEST_SUBJECTS
)

if len(subjects) != expected_subjects:

    raise RuntimeError(
        f"Expected {expected_subjects} subjects, "
        f"but found {len(subjects)}."
    )


# ============================================================
# SUBJECT BEHAVIOR COUNTS
# ============================================================

subject_stats = {}

for subject in subjects:

    counts = {
        "normal": 0,
        "talking": 0,
        "yawning": 0,
    }

    for item in subject_videos[subject]:
        counts[item["behavior"]] += 1

    subject_stats[subject] = counts


# ============================================================
# FIND A BALANCED SUBJECT SPLIT
# ============================================================

# Target global behavior proportions.
total_behavior = {
    "normal": sum(
        subject_stats[s]["normal"] for s in subjects
    ),
    "talking": sum(
        subject_stats[s]["talking"] for s in subjects
    ),
    "yawning": sum(
        subject_stats[s]["yawning"] for s in subjects
    ),
}

total_videos = sum(total_behavior.values())

target_ratio = {
    key: total_behavior[key] / total_videos
    for key in total_behavior
}


def split_score(selected_subjects):

    counts = {
        "normal": 0,
        "talking": 0,
        "yawning": 0,
    }

    for subject in selected_subjects:

        for behavior in counts:
            counts[behavior] += subject_stats[subject][behavior]

    total = sum(counts.values())

    if total == 0:
        return float("inf")

    score = 0

    for behavior in counts:

        actual_ratio = counts[behavior] / total

        score += abs(
            actual_ratio - target_ratio[behavior]
        )

    return score


# ============================================================
# RANDOMIZED SEARCH
# ============================================================

random.seed(RANDOM_SEED)

best_split = None
best_score = float("inf")

for _ in range(10000):

    shuffled = subjects.copy()
    random.shuffle(shuffled)

    train_subjects = shuffled[:TRAIN_SUBJECTS]
    val_subjects = shuffled[
        TRAIN_SUBJECTS:
        TRAIN_SUBJECTS + VAL_SUBJECTS
    ]
    test_subjects = shuffled[
        TRAIN_SUBJECTS + VAL_SUBJECTS:
    ]

    score = (
        split_score(train_subjects)
        + split_score(val_subjects)
        + split_score(test_subjects)
    )

    if score < best_score:

        best_score = score

        best_split = {
            "train": sorted(train_subjects),
            "val": sorted(val_subjects),
            "test": sorted(test_subjects),
        }


train_subjects = best_split["train"]
val_subjects = best_split["val"]
test_subjects = best_split["test"]


# ============================================================
# VERIFY NO SUBJECT LEAKAGE
# ============================================================

train_set = set(train_subjects)
val_set = set(val_subjects)
test_set = set(test_subjects)

assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)

assert (
    len(train_set)
    + len(val_set)
    + len(test_set)
    == 47
)


# ============================================================
# PRINT SUBJECT SPLIT
# ============================================================

print("\n" + "-" * 80)
print("SUBJECT SPLIT")
print("-" * 80)

print(f"\nTRAIN ({len(train_subjects)} subjects):")
print(train_subjects)

print(f"\nVALIDATION ({len(val_subjects)} subjects):")
print(val_subjects)

print(f"\nTEST ({len(test_subjects)} subjects):")
print(test_subjects)


# ============================================================
# SPLIT STATISTICS
# ============================================================

splits = {
    "train": train_subjects,
    "val": val_subjects,
    "test": test_subjects,
}


print("\n" + "=" * 80)
print("SPLIT BEHAVIOR DISTRIBUTION")
print("=" * 80)

manifest_rows = []


for split_name, split_subjects in splits.items():

    counts = {
        "normal": 0,
        "talking": 0,
        "yawning": 0,
    }

    video_count = 0

    for subject in split_subjects:

        for item in subject_videos[subject]:

            counts[item["behavior"]] += 1
            video_count += 1

            manifest_rows.append(
                {
                    "split": split_name,
                    "subject_id": subject,
                    "behavior": item["behavior"],
                    "filename": item["filename"],
                    "video_path": str(item["path"]),
                }
            )

    print(f"\n{split_name.upper()}")
    print(f"  Subjects: {len(split_subjects)}")
    print(f"  Videos:   {video_count}")
    print(f"  Normal:   {counts['normal']}")
    print(f"  Talking:  {counts['talking']}")
    print(f"  Yawning:  {counts['yawning']}")


# ============================================================
# SAVE MANIFEST
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    MANIFEST_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "subject_id",
            "behavior",
            "filename",
            "video_path",
        ]
    )

    writer.writeheader()
    writer.writerows(manifest_rows)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

print(f"\nTotal manifest rows: {len(manifest_rows)}")

assert len(manifest_rows) == 320

split_counts = defaultdict(int)

for row in manifest_rows:
    split_counts[row["split"]] += 1

print("\nVideos per split:")

for split_name in ["train", "val", "test"]:
    print(
        f"  {split_name:5s}: "
        f"{split_counts[split_name]}"
    )

print("\nSubject leakage check: PASSED")
print("Video count check: PASSED")
print(f"\nManifest saved to:")
print(MANIFEST_PATH)

print("\n" + "=" * 80)
print("SPLIT CREATION COMPLETE")
print("=" * 80)