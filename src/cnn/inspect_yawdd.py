from pathlib import Path
from collections import Counter, defaultdict
import cv2


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
# FIND VIDEOS
# ============================================================

videos = [
    p for p in YAWDD_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
]

print("=" * 70)
print("                    YawDD DATASET AUDIT")
print("=" * 70)

print(f"\nDataset root:")
print(YAWDD_ROOT)

print(f"\nTotal videos found: {len(videos)}")


# ============================================================
# BASIC COUNTS
# ============================================================

mirror_videos = []
dash_videos = []

for video in videos:
    parts = [part.lower() for part in video.parts]

    if "mirror" in parts:
        mirror_videos.append(video)

    elif "dash" in parts:
        dash_videos.append(video)


print("\nCamera configuration:")
print(f"  Mirror videos: {len(mirror_videos)}")
print(f"  Dash videos:   {len(dash_videos)}")


# ============================================================
# MIRROR BEHAVIOR COUNTS
# ============================================================

behavior_counts = Counter()
subject_videos = defaultdict(list)

for video in mirror_videos:

    name = video.stem.lower()

    if "-yawning" in name:
        behavior = "yawning"

    elif "-talking" in name:
        behavior = "talking"

    elif "-normal" in name:
        behavior = "normal"

    else:
        behavior = "unknown"

    behavior_counts[behavior] += 1

    # Subject ID = first part of filename
    # Example:
    # 1-FemaleNoGlasses-Normal
    first_part = video.stem.split("-")[0]

    if first_part.isdigit():
        subject_id = int(first_part)
        subject_videos[subject_id].append(video.name)


print("\nMirror behavior counts:")

for behavior in [
    "normal",
    "talking",
    "yawning",
    "unknown"
]:
    print(f"  {behavior:10s}: {behavior_counts[behavior]}")


# ============================================================
# SUBJECT SUMMARY
# ============================================================

print("\nMirror subjects:")
print(f"  Unique subjects: {len(subject_videos)}")

for subject_id in sorted(subject_videos):
    print(
        f"  Subject {subject_id:02d}: "
        f"{len(subject_videos[subject_id])} videos"
    )


# ============================================================
# VIDEO METADATA
# ============================================================

print("\n" + "=" * 70)
print("VIDEO METADATA SAMPLE")
print("=" * 70)

# Inspect first 10 videos only
for video in videos[:10]:

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        print(f"\nCould not open: {video.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    print(f"\nFile: {video.name}")
    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {int(frame_count)}")
    print(f"  Duration   : {duration:.2f} sec")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print("\nRecommended initial CNN source:")
print("  Mirror subset")

print("\nRecommended labels:")
print("  NON_YAWNING = Normal + Talking")
print("  YAWNING     = Yawning")

print("\nNo frames were extracted.")
print("No files were modified.")