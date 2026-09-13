from pathlib import Path
from collections import defaultdict


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

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


# ============================================================
# FIND MIRROR VIDEOS
# ============================================================

videos = [
    p for p in YAWDD_ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in VIDEO_EXTENSIONS
    and "Mirror" in p.parts
]


# ============================================================
# COLLECT SUBJECT / BEHAVIOR INFORMATION
# ============================================================

subject_data = defaultdict(lambda: defaultdict(list))

for video in videos:

    filename = video.stem

    parts = filename.split("-")

    # Example:
    # 1-FemaleNoGlasses-Normal
    # 10-MaleSunGlasses-Yawning

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
        behavior = "unknown"

    subject_data[subject_id][behavior].append(filename)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("=" * 80)
print("                 YawDD SUBJECT / BEHAVIOR AUDIT")
print("=" * 80)

print(f"\nDataset root:")
print(YAWDD_ROOT)

print(f"\nMirror videos found: {len(videos)}")
print(f"Unique subjects:     {len(subject_data)}")


print("\n" + "-" * 80)
print(
    f"{'Subject':<10}"
    f"{'Normal':<10}"
    f"{'Talking':<10}"
    f"{'Yawning':<10}"
    f"{'Total':<10}"
)
print("-" * 80)


total_normal = 0
total_talking = 0
total_yawning = 0

for subject_id in sorted(subject_data):

    normal = len(subject_data[subject_id]["normal"])
    talking = len(subject_data[subject_id]["talking"])
    yawning = len(subject_data[subject_id]["yawning"])

    total = normal + talking + yawning

    total_normal += normal
    total_talking += talking
    total_yawning += yawning

    print(
        f"{subject_id:<10}"
        f"{normal:<10}"
        f"{talking:<10}"
        f"{yawning:<10}"
        f"{total:<10}"
    )


print("-" * 80)

print(
    f"{'TOTAL':<10}"
    f"{total_normal:<10}"
    f"{total_talking:<10}"
    f"{total_yawning:<10}"
    f"{total_normal + total_talking + total_yawning:<10}"
)


# ============================================================
# CHECK SUBJECT COMPLETENESS
# ============================================================

print("\n" + "=" * 80)
print("SUBJECT COMPLETENESS")
print("=" * 80)

complete_subjects = []
incomplete_subjects = []

for subject_id in sorted(subject_data):

    behaviors = subject_data[subject_id]

    has_normal = len(behaviors["normal"]) > 0
    has_talking = len(behaviors["talking"]) > 0
    has_yawning = len(behaviors["yawning"]) > 0

    if has_normal and has_talking and has_yawning:
        complete_subjects.append(subject_id)

    else:
        incomplete_subjects.append(
            (
                subject_id,
                has_normal,
                has_talking,
                has_yawning
            )
        )


print(f"\nSubjects with all 3 behaviors: {len(complete_subjects)}")
print(f"Subjects with missing behavior: {len(incomplete_subjects)}")

if incomplete_subjects:

    print("\nIncomplete subjects:")

    for (
        subject_id,
        has_normal,
        has_talking,
        has_yawning
    ) in incomplete_subjects:

        print(
            f"  Subject {subject_id:02d}: "
            f"Normal={has_normal}, "
            f"Talking={has_talking}, "
            f"Yawning={has_yawning}"
        )


print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)