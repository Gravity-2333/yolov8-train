# DENTEX v4 Region NWD-CIoU Quick Ablation Report

## Summary

This report records the source-level NWD-CIoU quick ablation on DENTEX v4_region.

- Experiment branch: `exp/dentex-v4-nwd-ciou`
- Implementation commit: `b16f5311c Implement DENTEX v4 NWD-CIoU loss`
- Dataset: `ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml`
- Model: `yolov8n.pt`
- Image size: `1280`
- Epochs: `50`
- Batch: `16`
- Workers: `24`
- BBox loss type: `nwd_ciou`
- `nwd-ratio`: `0.5`
- `nwd-constant`: `12.8`
- Test split: not used

The quick run completed 50 epochs. There was no OOM, NaN, path error, class error, backward error, or channel mismatch.

## Source Changes

Modified files:

- `ultralytics/utils/metrics.py`
  - Added `bbox_nwd_similarity()`.
  - Kept `bbox_iou()` default behavior unchanged.
- `ultralytics/utils/loss.py`
  - Added opt-in `nwd_ciou` path in `BboxLoss`.
  - Default path remains CIoU.
- `scripts/train_dental_ablation.py`
  - Added `--bbox-loss-type nwd_ciou`.
  - Added `--nwd-ratio`.
  - Added `--nwd-constant`.
  - Writes NWD parameters to `run_metadata.yaml`.
  - Clears/overwrites NWD environment variables per run to avoid cross-run contamination.

## NWD-CIoU Formula

For box similarity:

`mixed = (1 - nwd_ratio) * ciou + nwd_ratio * nwd`

For the first quick ablation:

`mixed = 0.5 * ciou + 0.5 * nwd`

The bbox loss keeps the same weighted YOLOv8 form:

`loss_iou = ((1.0 - mixed) * weight).sum() / target_scores_sum`

NWD is implemented as a similarity where values closer to `1` mean boxes are closer. The implementation is opt-in. `--bbox-loss-type ciou` uses the original CIoU path.

## Pre-Training Checks

All checks passed:

- `python -m py_compile ultralytics/utils/metrics.py`
- `python -m py_compile ultralytics/utils/loss.py`
- `python -m py_compile scripts/train_dental_ablation.py`
- Tensor check for `bbox_nwd_similarity()`
  - same box: `0.999975`
  - shifted box: `0.895399`
  - far box: `0.109733`
  - finite gradients verified
- Official model build check:
  - `yolov8n.yaml`
  - `yolov8s.yaml`
  - `yolov8m.yaml`

## Smoke And Sanity

### CIoU Smoke

- Run: `dentex_v4_region_yolov8n_ciou_640_smoke_after_nwd_impl_20260523_174953_b16f531`
- Epochs: `1`
- Image size: `640`
- Batch: `8`
- Result: passed
- Artifacts present:
  - `results.csv`
  - `run_metadata.yaml`
  - `weights/best.pt`
  - `weights/last.pt`

### NWD-CIoU Sanity

- Run: `dentex_v4_region_yolov8n_nwd_ciou_640_sanity_20260523_175102_b16f531`
- Epochs: `1`
- Image size: `640`
- Batch: `8`
- Result: passed
- Artifacts present:
  - `results.csv`
  - `run_metadata.yaml`
  - `weights/best.pt`
  - `weights/last.pt`

## Quick Run

- Run: `dentex_v4_region_yolov8n_nwd_ciou_1280_quick_b16_r05_c128_20260523_175211_b16f531`
- Run path: `runs/detect/dentex_v4_region_nwd_quick/dentex_v4_region_yolov8n_nwd_ciou_1280_quick_b16_r05_c128_20260523_175211_b16f531`
- Best weights: `runs/detect/dentex_v4_region_nwd_quick/dentex_v4_region_yolov8n_nwd_ciou_1280_quick_b16_r05_c128_20260523_175211_b16f531/weights/best.pt`
- Completed epochs: `50`
- Best epoch by training `results.csv`: `27`
- GPU peak from training log: about `7.99 GB`
- Last epoch metrics from `results.csv`: P `0.5967`, R `0.6550`, mAP50 `0.6204`, mAP50-95 `0.3651`
- Best epoch metrics from `results.csv`: P `0.6339`, R `0.6523`, mAP50 `0.6662`, mAP50-95 `0.3864`

## Unified Val Compare

Comparison output:

`reports/ablation/dentex_v4_region_nwd_ciou_quick_compare_20260523_181558/compare_20260523_181602_b16f531`

Both models were validated on the same v4 val split with `imgsz=1280`, `conf=0.001`, and `iou=0.7`.

| Model | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n 1280 quick b16 | 0.5691 | 0.6615 | 0.6479 | 0.3859 |
| YOLOv8n NWD-CIoU 1280 quick b16 | 0.6335 | 0.6502 | 0.6640 | 0.3868 |

## Per-Class Results

| Model | Class | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLOv8n 1280 quick b16 | Caries | 0.5116 | 0.5000 | 0.5155 | 0.3428 |
| YOLOv8n 1280 quick b16 | Periapical_Lesion | 0.6199 | 0.5812 | 0.6095 | 0.2523 |
| YOLOv8n 1280 quick b16 | Impacted | 0.5759 | 0.9032 | 0.8186 | 0.5627 |
| YOLOv8n NWD-CIoU 1280 quick b16 | Caries | 0.5787 | 0.5339 | 0.5633 | 0.3536 |
| YOLOv8n NWD-CIoU 1280 quick b16 | Periapical_Lesion | 0.6279 | 0.5459 | 0.5846 | 0.2520 |
| YOLOv8n NWD-CIoU 1280 quick b16 | Impacted | 0.6939 | 0.8710 | 0.8441 | 0.5549 |

## Delta Versus 1280 Quick Baseline

- Overall P: `+0.0643`
- Overall R: `-0.0112`
- Overall mAP50: `+0.0161`
- Overall mAP50-95: `+0.0009`
- Caries recall: `+0.0339`
- Caries mAP50-95: `+0.0108`
- Periapical_Lesion precision: `+0.0080`
- Periapical_Lesion recall: `-0.0353`
- Periapical_Lesion mAP50: `-0.0249`
- Periapical_Lesion mAP50-95: `-0.0003`
- Impacted recall: `-0.0323`
- Impacted mAP50: `+0.0255`
- Impacted mAP50-95: `-0.0078`

## Context Versus P2 And 1536 Quick

P2 quick was rejected because it increased computation and reduced the key Periapical_Lesion recall.

1536 quick improved overall mAP50-95 under unified 1536 validation, but Periapical_Lesion recall fell and class 1 mAP50-95 was essentially unchanged.

NWD-CIoU quick behaves similarly: it slightly improves overall mAP50 and precision, but it does not improve the key class 1 strict localization metric and reduces class 1 recall.

## Recommendation

Do not run NWD-CIoU full training.

The NWD-CIoU quick result does not satisfy the intended success criteria:

- Periapical_Lesion mAP50-95 did not improve: `0.2523 -> 0.2520`.
- Periapical_Lesion recall dropped: `0.5812 -> 0.5459`.
- Periapical_Lesion mAP50 dropped: `0.6095 -> 0.5846`.
- Overall mAP50-95 improved by only `+0.0009`, which is not meaningful enough to justify a full run.

The current main candidate should remain:

`v4_region YOLOv8n 1280 full best.pt`

No test split was used in this experiment. Do not merge this experimental branch into `feat/dental-neckplus-v1` unless there is a later reason to preserve NWD-CIoU as an optional research feature.
