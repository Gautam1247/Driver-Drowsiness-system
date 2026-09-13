from pathlib import Path
import cv2
import matplotlib.pyplot as plt


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

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "yawdd_samples"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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


# ============================================================
# SELECT REPRESENTATIVE VIDEOS
# ============================================================

selected = []

for behavior in ["Normal", "Talking", "Yawning"]:

    matches = [
        video
        for video in videos
        if f"-{behavior.lower()}" in video.stem.lower()
    ]

    # Select a few different subjects
    selected.extend(matches[:3])


# ============================================================
# EXTRACT REPRESENTATIVE FRAMES
# ============================================================

for video in selected:

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        print(f"Could not open: {video}")
        continue

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Take frames from:
    # 25%, 50%, 75% of video
    positions = [
        0.25,
        0.50,
        0.75
    ]

    frames = []

    for position in positions:

        frame_number = int(
            total_frames * position
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(
            (
                position,
                frame
            )
        )

    cap.release()

    if not frames:
        continue


    # ========================================================
    # CREATE CONTACT SHEET
    # ========================================================

    fig, axes = plt.subplots(
        1,
        len(frames),
        figsize=(15, 5)
    )

    if len(frames) == 1:
        axes = [axes]

    for ax, (position, frame) in zip(
        axes,
        frames
    ):

        ax.imshow(frame)

        ax.set_title(
            f"{position * 100:.0f}%"
        )

        ax.axis("off")


    fig.suptitle(
        f"{video.name}\n"
        f"{fps:.2f} FPS | {total_frames} frames",
        fontsize=12
    )

    plt.tight_layout()

    output_name = (
        video.stem
        + "_samples.jpg"
    )

    output_path = (
        OUTPUT_DIR
        / output_name
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


print("\n" + "=" * 70)
print("FRAME INSPECTION COMPLETE")
print("=" * 70)

print(f"\nSample images saved to:")
print(OUTPUT_DIR)