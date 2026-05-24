#!/usr/bin/env python
"""Validate multiple dental YOLOv8 weights on one dataset split."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


def git_short() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dental YOLOv8 ablation weights.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/val/dental_neckplus")
    parser.add_argument("--out-dir", default="reports/ablation")
    return parser.parse_args()


def scalar(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def model_stats(model: YOLO, weights_path: Path, imgsz: int) -> tuple[int, float, float]:
    params = sum(p.numel() for p in model.model.parameters()) if hasattr(model, "model") else 0
    gflops = scalar(get_flops(model.model, imgsz=imgsz)) if hasattr(model, "model") else 0.0
    size_mb = weights_path.stat().st_size / (1024 * 1024) if weights_path.exists() else 0.0
    return params, gflops, size_mb


def main() -> None:
    args = parse_args()
    if args.names and len(args.names) != len(args.weights):
        raise SystemExit("--names length must match --weights length.")
    names = args.names or [Path(w).stem for w in args.weights]
    out_dir = Path(args.out_dir) / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{git_short()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, per_class_rows = [], []
    for experiment, weights in zip(names, args.weights):
        print(f"Validating {experiment}: {weights}")
        model = YOLO(weights)
        val_kwargs = {
            "data": args.data,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "split": args.split,
            "project": args.project,
            "name": experiment,
            "exist_ok": True,
            "plots": False,
        }
        if args.device != "":
            val_kwargs["device"] = args.device
        metrics = model.val(**val_kwargs)
        params, gflops, size_mb = model_stats(model, Path(weights), args.imgsz)
        speed = getattr(metrics, "speed", {}) or {}
        mean = metrics.mean_results() if hasattr(metrics, "mean_results") else [0, 0, 0, 0]
        summary_rows.append(
            {
                "experiment": experiment,
                "weights": weights,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "split": args.split,
                "precision": scalar(mean[0]),
                "recall": scalar(mean[1]),
                "mAP50": scalar(mean[2]),
                "mAP50-95": scalar(mean[3]),
                "params": params,
                "GFLOPs": gflops,
                "model_size_MB": size_mb,
                "speed_preprocess": scalar(speed.get("preprocess", 0.0)),
                "speed_inference": scalar(speed.get("inference", 0.0)),
                "speed_postprocess": scalar(speed.get("postprocess", 0.0)),
            }
        )
        class_names = getattr(metrics, "names", getattr(model, "names", {})) or {}
        for cls in sorted(class_names):
            try:
                p, r, map50, map5095 = metrics.class_result(int(cls))
            except Exception:
                p, r, map50, map5095 = 0, 0, 0, 0
            per_class_rows.append(
                {
                    "experiment": experiment,
                    "split": args.split,
                    "class_id": int(cls),
                    "class_name": class_names[cls],
                    "precision": scalar(p),
                    "recall": scalar(r),
                    "mAP50": scalar(map50),
                    "mAP50-95": scalar(map5095),
                }
            )

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()) if per_class_rows else ["experiment", "split", "class_id", "class_name", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        writer.writerows(per_class_rows)

    best = max(summary_rows, key=lambda r: (r["mAP50-95"], r["recall"], r["mAP50"]))
    lines = [
        "# Dental Ablation Comparison",
        "",
        f"- Split: `{args.split}`",
        "",
        "| experiment | P | R | mAP50 | mAP50-95 | GFLOPs | size MB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['experiment']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['mAP50']:.4f} | "
            f"{row['mAP50-95']:.4f} | {row['GFLOPs']:.3f} | {row['model_size_MB']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            f"- Results use the `{args.split}` split.",
            "- Test split should be used only for locked final confirmation.",
            "- First-stage small/medium/large analysis should be read together with `dataset_audit_dental.py` outputs.",
        ]
    )
    (out_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    recommendation = [
        "# Best Model Recommendation",
        "",
        f"Recommended first candidate: `{best['experiment']}`.",
        "",
        "Selection priority: mAP50-95, recall, per-class balance, model size, and speed.",
        "Review `per_class.csv` before starting long training, especially for weak lesion classes.",
    ]
    (out_dir / "best_model_recommendation.md").write_text("\n".join(recommendation) + "\n", encoding="utf-8")
    print(f"Comparison complete: {out_dir}")


if __name__ == "__main__":
    main()
