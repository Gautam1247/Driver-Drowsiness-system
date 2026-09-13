import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT /
    "datasets" /
    "mouth" /
    "cnn_dataset"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "evaluation" /
    "mouth_dataset_samples"
)

RANDOM_SEED = 42
SAMPLES_PER_CATEGORY = 12


def collect_images(directory):

    extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    return [
        path
        for path in directory.rglob("*")
        if path.suffix.lower() in extensions
    ]


def create_contact_sheet(
    image_paths,
    title,
    output_path,
    cols=4
):

    rows = (len(image_paths) + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, 3 * rows)
    )

    # Make axes iterable even for one row.
    if rows == 1:
        axes = [axes]

    axes = [
        ax
        for row in axes
        for ax in row
    ]

    for ax in axes:
        ax.axis("off")

    for ax, image_path in zip(
        axes,
        image_paths
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        ax.imshow(image)
        ax.set_title(
            image_path.name[:22],
            fontsize=7
        )
        ax.axis("off")

    fig.suptitle(
        title,
        fontsize=16
    )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():

    random.seed(RANDOM_SEED)

    print("=" * 70)
    print("MOUTH DATASET VISUAL INSPECTION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    categories = [
        ("train", "yawning"),
        ("train", "non_yawning"),
        ("val", "yawning"),
        ("val", "non_yawning"),
        ("test", "yawning"),
        ("test", "non_yawning"),
    ]

    for split, class_name in categories:

        directory = (
            DATASET_DIR /
            split /
            class_name
        )

        images = collect_images(
            directory
        )

        print(
            f"\n{split.upper():5s} | "
            f"{class_name:12s} | "
            f"{len(images):4d} images"
        )

        if not images:
            print("WARNING: no images found.")
            continue

        sample_count = min(
            SAMPLES_PER_CATEGORY,
            len(images)
        )

        samples = random.sample(
            images,
            sample_count
        )

        output_path = (
            OUTPUT_DIR /
            f"{split}_{class_name}.png"
        )

        create_contact_sheet(
            samples,
            f"{split.upper()} — {class_name}",
            output_path
        )

        print(
            f"Saved: {output_path}"
        )

    print("\n" + "=" * 70)
    print("INSPECTION CONTACT SHEETS CREATED")
    print("=" * 70)


if __name__ == "__main__":
    main()