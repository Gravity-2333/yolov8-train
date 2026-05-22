"""Create a repeated-image train list for small dental lesion classes.

The script reads a YOLO detection dataset and writes a text file containing
image paths. Images are repeated by priority:
Periapical Lesion > Impacted > Caries.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="YOLO dataset yaml.")
    parser.add_argument("--out", required=True, help="Output train txt path.")
    parser.add_argument("--periapical-repeat", type=int, default=4)
    parser.add_argument("--impacted-repeat", type=int, default=2)
    parser.add_argument("--caries-repeat", type=int, default=1)
    return parser.parse_args()


def resolve_split(root: Path, value: str) -> Path:
    split = Path(value)
    return split if split.is_absolute() else root / split


def image_to_label(root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(root)
    parts = list(rel.parts)
    if parts[0] != "images":
        raise ValueError(f"Expected image under images/: {image_path}")
    parts[0] = "labels"
    return root.joinpath(*parts).with_suffix(".txt")


def label_classes(label_path: Path) -> set[int]:
    classes: set[int] = set()
    if not label_path.exists():
        return classes
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        try:
            classes.add(int(float(fields[0])))
        except ValueError:
            continue
    return classes


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data["path"]).expanduser()
    train_dir = resolve_split(root, data["train"])
    if train_dir.is_file():
        raise ValueError("This script expects the source train split to be an image directory.")

    image_paths = sorted(p for p in train_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    repeated: list[Path] = []
    image_priority_counts: Counter[str] = Counter()
    class_image_counts: Counter[int] = Counter()
    class_box_counts: Counter[int] = Counter()

    for image_path in image_paths:
        label_path = image_to_label(root, image_path)
        classes = label_classes(label_path)
        for class_id in classes:
            class_image_counts[class_id] += 1
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                fields = line.strip().split()
                if fields:
                    try:
                        class_box_counts[int(float(fields[0]))] += 1
                    except ValueError:
                        pass

        if 1 in classes:
            repeat = args.periapical_repeat
            priority = "periapical"
        elif 2 in classes:
            repeat = args.impacted_repeat
            priority = "impacted"
        else:
            repeat = args.caries_repeat
            priority = "caries_or_empty"
        repeated.extend([image_path.resolve()] * repeat)
        image_priority_counts[priority] += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(p) for p in repeated) + "\n", encoding="utf-8")

    print(f"source_images: {len(image_paths)}")
    print(f"balanced_entries: {len(repeated)}")
    print(f"output: {out_path}")
    print("priority_image_counts:")
    for key, value in sorted(image_priority_counts.items()):
        print(f"  {key}: {value}")
    print("class_image_counts:")
    for key, value in sorted(class_image_counts.items()):
        print(f"  {key}: {value}")
    print("class_box_counts:")
    for key, value in sorted(class_box_counts.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
