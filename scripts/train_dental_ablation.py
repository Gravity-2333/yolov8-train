#!/usr/bin/env python
"""Train dental YOLOv8 ablations with reproducible run metadata."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def git_dirty() -> bool:
    return bool(git_value(["status", "--short"], default=""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dental YOLOv8 ablation.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", default="")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="dental_ablation")
    parser.add_argument("--project", default="runs/detect/dental_neckplus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bbox-loss-type", default="ciou")
    parser.add_argument("--cache", default=None, help="Ultralytics cache option, e.g. ram, disk, True, False.")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def normalize_cache(value: str | None):
    if value is None:
        return None
    lowered = str(value).lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no", "none"}:
        return False
    return value


def main() -> None:
    args = parse_args()
    if args.bbox_loss_type != "ciou":
        raise SystemExit("第一阶段尚未接入该 bbox loss，请使用 --bbox-loss-type ciou。")

    commit = git_value(["rev-parse", "--short", "HEAD"])
    branch = git_value(["branch", "--show-current"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.name}_{timestamp}_{commit}"
    run_dir = Path(args.project) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": run_name,
        "seed": args.seed,
        # The timestamped name is unique; exist_ok=True keeps Ultralytics from incrementing the pre-created metadata dir.
        "exist_ok": True,
        "cos_lr": True,
        "close_mosaic": 20,
        "patience": 100,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.1,
        "hsv_v": 0.15,
        "degrees": 5.0,
        "translate": 0.05,
        "scale": 0.3,
    }
    if args.device != "":
        train_kwargs["device"] = args.device
    cache = normalize_cache(args.cache)
    if cache is not None:
        train_kwargs["cache"] = cache

    metadata = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "branch": branch,
        "commit": commit,
        "dirty": git_dirty(),
        "command_line": " ".join(shlex.quote(v) for v in __import__("sys").argv),
        "data_yaml": args.data,
        "model_yaml_or_weights": args.model,
        "weights": args.weights,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "bbox_loss_type": args.bbox_loss_type,
        "exist_ok_requested": args.exist_ok,
        "train_kwargs": train_kwargs,
        "pretrained_loading": {
            "requested_weights": args.weights,
            "status": "not_requested",
            "note": "Ultralytics may skip shape-mismatched layers; confirm exact transfer counts in the training log.",
        },
    }

    print("Dental ablation training")
    for key in ("branch", "commit", "dirty", "data_yaml", "model_yaml_or_weights", "weights", "imgsz", "epochs", "batch", "device", "workers", "bbox_loss_type"):
        print(f"{key}: {metadata[key]}")

    model = YOLO(args.model)
    if args.weights and args.weights != args.model:
        try:
            model.load(args.weights)
            metadata["pretrained_loading"]["status"] = "loaded_via_ultralytics_model_load"
        except Exception as exc:
            metadata["pretrained_loading"]["status"] = "failed"
            metadata["pretrained_loading"]["error"] = str(exc)
            (run_dir / "run_metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8")
            raise
    elif args.weights and args.weights == args.model:
        metadata["pretrained_loading"]["status"] = "model_argument_is_pretrained_weights"

    (run_dir / "run_metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
