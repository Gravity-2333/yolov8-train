# DENTEX v4 Region EMA-P3Lite Quick Ablation Report

## Summary

This report records the YOLOv8n + EMA-P3Lite quick ablation on DENTEX v4_region.

- Experiment branch: `exp/dentex-v4-ema-p3lite`
- Implementation commit: `f6f075b89 Implement DENTEX v4 EMA-P3Lite model`
- Dataset: `ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml`
- Model YAML: `ultralytics/cfg/models/v8/yolov8n_dental_ema_p3lite.yaml`
- Image size: `1280`
- Epochs: `50`
- Batch: `16`
- Workers: `24`
- Loss: original CIoU
- Test split: not used

The quick run completed 50 epochs with no OOM, NaN, channel mismatch, Detect shape error, path error, or class error.

## Source Changes

Modified files:

- `ultralytics/nn/modules/dental_neck.py`
  - Added `DentalEMA`, a grouped lightweight efficient multi-scale attention block.
- `ultralytics/nn/modules/__init__.py`
  - Exported `DentalEMA`.
- `ultralytics/nn/tasks.py`
  - Registered `DentalEMA` in `parse_model()` with the same single-input channel-preserving handling as `DentalECA`.
- `ultralytics/cfg/models/v8/yolov8n_dental_ema_p3lite.yaml`
  - Added a YOLOv8n variant with `DentalEMA` only on the P3 branch before Detect.

No loss, dataset, P2 head, SAHI, balanced sampling, or test evaluation was used.

## Model Structure

EMA-P3Lite keeps the standard YOLOv8n P3/P4/P5 Detect head. It adds only one `DentalEMA` block at the P3 feature map before Detect.

Build summary:

- Unfused build: `3,157,872` parameters, `8.9` GFLOPs
- Fused val summary: `3,006,905` parameters, `32.615` GFLOPs at `imgsz=1280`
- Baseline fused val summary: `3,006,233` parameters, `32.349` GFLOPs at `imgsz=1280`

The added compute is small.

## Checks

- Local py_compile: passed
- Local model build: passed
- Server py_compile: passed
- Server model build: passed
- 1 epoch sanity: passed

Sanity run:

`runs/detect/dentex_v4_region_ema_p3lite_sanity/dentex_v4_region_yolov8n_ema_p3lite_640_sanity_20260523_182633_f6f075b`

## Quick Run

- Run: `dentex_v4_region_yolov8n_ema_p3lite_1280_quick_b16_20260523_182735_f6f075b`
- Run path: `runs/detect/dentex_v4_region_ema_p3lite_quick/dentex_v4_region_yolov8n_ema_p3lite_1280_quick_b16_20260523_182735_f6f075b`
- Best weights: `runs/detect/dentex_v4_region_ema_p3lite_quick/dentex_v4_region_yolov8n_ema_p3lite_1280_quick_b16_20260523_182735_f6f075b/weights/best.pt`
- Completed epochs: `50`
- Best epoch by training `results.csv`: `21`
- GPU peak from training log: about `8.32 GB`
- Best epoch metrics from `results.csv`: P `0.5631`, R `0.6440`, mAP50 `0.6248`, mAP50-95 `0.3686`
- Last epoch metrics from `results.csv`: P `0.5863`, R `0.6782`, mAP50 `0.5859`, mAP50-95 `0.3512`

## Unified Val Compare

Comparison output:

`reports/ablation/dentex_v4_region_ema_p3lite_quick_compare_20260523_185154/compare_20260523_185157_f6f075b`

Both models were validated on the same v4 val split with `imgsz=1280`, `conf=0.001`, and `iou=0.7`.

| Model | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n 1280 quick b16 | 0.5691 | 0.6615 | 0.6479 | 0.3859 |
| YOLOv8n EMA-P3Lite 1280 quick b16 | 0.5634 | 0.6448 | 0.6261 | 0.3698 |

## Per-Class Results

| Model | Class | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLOv8n 1280 quick b16 | Caries | 0.5116 | 0.5000 | 0.5155 | 0.3428 |
| YOLOv8n 1280 quick b16 | Periapical_Lesion | 0.6199 | 0.5812 | 0.6095 | 0.2523 |
| YOLOv8n 1280 quick b16 | Impacted | 0.5759 | 0.9032 | 0.8186 | 0.5627 |
| YOLOv8n EMA-P3Lite 1280 quick b16 | Caries | 0.4750 | 0.5367 | 0.5237 | 0.3503 |
| YOLOv8n EMA-P3Lite 1280 quick b16 | Periapical_Lesion | 0.6549 | 0.5268 | 0.5775 | 0.2529 |
| YOLOv8n EMA-P3Lite 1280 quick b16 | Impacted | 0.5602 | 0.8710 | 0.7771 | 0.5061 |

## Delta Versus 1280 Quick Baseline

- Overall P: `-0.0058`
- Overall R: `-0.0166`
- Overall mAP50: `-0.0218`
- Overall mAP50-95: `-0.0162`
- Caries recall: `+0.0367`
- Caries mAP50-95: `+0.0075`
- Periapical_Lesion precision: `+0.0350`
- Periapical_Lesion recall: `-0.0544`
- Periapical_Lesion mAP50: `-0.0320`
- Periapical_Lesion mAP50-95: `+0.0006`
- Impacted recall: `-0.0323`
- Impacted mAP50: `-0.0415`
- Impacted mAP50-95: `-0.0566`

## Context Versus Prior Attempts

- P2 quick: rejected because Periapical_Lesion recall and Impacted recall dropped while compute increased.
- 1536 quick: improved overall mAP50-95 slightly, but Periapical_Lesion recall dropped and class 1 mAP50-95 was essentially unchanged.
- NWD-CIoU quick: overall mAP50-95 changed by only about `+0.0009`; Periapical_Lesion recall and mAP50 dropped.
- EMA-P3Lite quick: does not improve the target class enough and reduces overall metrics.

## Recommendation

Do not run EMA-P3Lite full training.

EMA-P3Lite does not meet the success criteria:

- Overall mAP50-95 drops from `0.3859` to `0.3698`.
- Periapical_Lesion recall drops from `0.5812` to `0.5268`.
- Periapical_Lesion mAP50 drops from `0.6095` to `0.5775`.
- Periapical_Lesion mAP50-95 changes only from `0.2523` to `0.2529`, which is too small to justify full training.
- Impacted degrades substantially, especially mAP50-95: `0.5627 -> 0.5061`.

The current main candidate should remain:

`v4_region YOLOv8n 1280 full best.pt`

No test split was used. Do not merge this experimental branch into `feat/dental-neckplus-v1` unless there is a later reason to preserve EMA-P3Lite as an optional research feature.
