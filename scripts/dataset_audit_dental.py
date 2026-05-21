#!/usr/bin/env python
"""Audit a YOLO dental lesion dataset and create lightweight visual checks."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a YOLO-format dental lesion dataset.")
    parser.add_argument("--data", required=True, help="Path to YOLO data yaml.")
    parser.add_argument("--out-dir", default="reports/dataset_audit", help="Directory for audit outputs.")
    parser.add_argument("--max-preview", type=int, default=20, help="Number of random GT visualizations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_data_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = Path(data.get("path", "."))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    data["_root"] = root
    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    data["names"] = {int(k): str(v) for k, v in names.items()}
    return data


def resolve_split(root: Path, split_value: str | list[str]) -> list[Path]:
    values = split_value if isinstance(split_value, list) else [split_value]
    paths = []
    for value in values:
        p = Path(value)
        paths.append(p if p.is_absolute() else root / p)
    return paths


def iter_images(paths: list[Path]) -> list[Path]:
    images = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            images.append(path)
        elif path.exists():
            images.extend(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return sorted(images)


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def bucket(area: float) -> str:
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def read_label(label_path: Path, names: dict[int, str]) -> tuple[list[tuple[int, float, float, float, float]], Counter, list[str]]:
    boxes = []
    errors = Counter()
    warnings = []
    if not label_path.exists():
        errors["missing_label"] += 1
        return boxes, errors, warnings
    lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        errors["empty_label"] += 1
        return boxes, errors, warnings
    for line_no, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            errors["bad_format"] += 1
            warnings.append(f"{label_path}:{line_no} expected 5 columns, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors["bad_format"] += 1
            warnings.append(f"{label_path}:{line_no} contains non-numeric values")
            continue
        if cls not in names:
            errors["bad_class"] += 1
            warnings.append(f"{label_path}:{line_no} class {cls} is outside names")
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errors["bad_bbox"] += 1
            warnings.append(f"{label_path}:{line_no} invalid bbox {[x, y, w, h]}")
            continue
        if x - w / 2 < -1e-6 or y - h / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y + h / 2 > 1 + 1e-6:
            errors["bbox_outside_image"] += 1
            warnings.append(f"{label_path}:{line_no} bbox extends outside image")
            continue
        boxes.append((cls, x, y, w, h))
    return boxes, errors, warnings


def draw_preview(image_path: Path, boxes: list[tuple[int, float, float, float, float]], names: dict[int, str], out_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    h, w = image.shape[:2]
    colors = [(40, 180, 255), (80, 220, 120), (255, 120, 60), (180, 120, 255)]
    for cls, x, y, bw, bh in boxes:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        color = colors[cls % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, names.get(cls, str(cls)), (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    data_path = Path(args.data)
    data = load_data_yaml(data_path)
    names = data["names"]
    out_dir = Path(args.out_dir)
    preview_dir = out_dir / "random_visual_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    class_rows, area_rows, report_rows = [], [], []
    warnings: list[str] = []
    preview_candidates = []
    totals = Counter()

    for split in ("train", "val", "test"):
        if split not in data:
            report_rows.append({"split": split, "exists": False})
            continue
        image_paths = iter_images(resolve_split(data["_root"], data[split]))
        split_counter = Counter()
        class_boxes = Counter()
        class_images = Counter()
        area_buckets = Counter()
        wh_area = []

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                split_counter["unreadable_image"] += 1
                warnings.append(f"Unreadable image: {image_path}")
                continue
            label_path = label_path_for(image_path)
            boxes, errors, label_warnings = read_label(label_path, names)
            split_counter.update(errors)
            warnings.extend(label_warnings[:20])
            split_counter["images"] += 1
            split_counter["labels"] += int(label_path.exists())
            split_counter["boxes"] += len(boxes)
            present = set()
            for cls, _, _, bw, bh in boxes:
                area = bw * bh
                class_boxes[cls] += 1
                present.add(cls)
                area_buckets[bucket(area)] += 1
                wh_area.append((bw, bh, area))
            for cls in present:
                class_images[cls] += 1
            if boxes:
                preview_candidates.append((split, image_path, boxes))

        totals.update({f"{split}_{k}": v for k, v in split_counter.items()})
        avg_boxes = split_counter["boxes"] / split_counter["images"] if split_counter["images"] else 0.0
        report_rows.append(
            {
                "split": split,
                "exists": True,
                "images": split_counter["images"],
                "labels": split_counter["labels"],
                "boxes": split_counter["boxes"],
                "empty_labels": split_counter["empty_label"],
                "missing_labels": split_counter["missing_label"],
                "bad_format": split_counter["bad_format"],
                "bad_class": split_counter["bad_class"],
                "bad_bbox": split_counter["bad_bbox"],
                "bbox_outside_image": split_counter["bbox_outside_image"],
                "avg_boxes_per_image": avg_boxes,
            }
        )
        for cls, name in names.items():
            class_rows.append([split, cls, name, class_boxes[cls], class_images[cls]])
        for size_name in ("small", "medium", "large"):
            area_rows.append([split, size_name, area_buckets[size_name]])
        if wh_area:
            widths, heights, areas = zip(*wh_area)
            area_rows.extend(
                [
                    [split, "width_mean", sum(widths) / len(widths)],
                    [split, "height_mean", sum(heights) / len(heights)],
                    [split, "area_mean", sum(areas) / len(areas)],
                ]
            )

    with (out_dir / "class_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_id", "class_name", "box_count", "image_count_with_class"])
        writer.writerows(class_rows)

    with (out_dir / "bbox_area_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "metric", "value"])
        writer.writerows(area_rows)

    random.shuffle(preview_candidates)
    for idx, (split, image_path, boxes) in enumerate(preview_candidates[: args.max_preview], start=1):
        draw_preview(image_path, boxes, names, preview_dir / f"{idx:03d}_{split}_{image_path.stem}.jpg")

    md = ["# Dental Dataset Audit", "", f"- data yaml: `{data_path}`", f"- dataset root: `{data['_root']}`", ""]
    md.append("## Split Summary")
    md.append("| split | images | labels | boxes | empty | missing | bad_format | bad_class | bad_bbox | outside | avg_boxes |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report_rows:
        if not row.get("exists"):
            md.append(f"| {row['split']} | missing split | | | | | | | | | |")
            continue
        md.append(
            f"| {row['split']} | {row['images']} | {row['labels']} | {row['boxes']} | {row['empty_labels']} | "
            f"{row['missing_labels']} | {row['bad_format']} | {row['bad_class']} | {row['bad_bbox']} | "
            f"{row['bbox_outside_image']} | {row['avg_boxes_per_image']:.3f} |"
        )
    md.extend(["", "## Warnings"])
    if warnings:
        md.extend(f"- {w}" for w in warnings[:200])
        if len(warnings) > 200:
            md.append(f"- ... truncated {len(warnings) - 200} additional warnings")
    else:
        md.append("- No warnings.")
    md.extend(["", "## Outputs", "- `class_distribution.csv`", "- `bbox_area_distribution.csv`", "- `random_visual_check/*.jpg`"])
    (out_dir / "dataset_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Audit complete: {out_dir}")


if __name__ == "__main__":
    main()
