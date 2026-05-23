# DENTEX v4 Region EMCA-Lite Quick Ablation Report

## Summary

This report records the YOLOv8n + EMCA-Lite quick ablation on DENTEX v4_region.

- Experiment branch: `exp/dentex-v4-emca-lite`
- Implementation commit: `3405a79d3 Implement DENTEX v4 EMCA-Lite model`
- Dataset: `ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml`
- Model YAML: `ultralytics/cfg/models/v8/yolov8n_dental_emca_lite.yaml`
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
  - Added `DentalEMCALite`, a lightweight mixed channel-spatial attention block.
- `ultralytics/nn/modules/__init__.py`
  - Exported `DentalEMCALite`.
- `ultralytics/nn/tasks.py`
  - Registered `DentalEMCALite` in `parse_model()` as a single-input channel-preserving module.
- `ultralytics/cfg/models/v8/yolov8n_dental_emca_lite.yaml`
  - Added a YOLOv8n variant with `DentalEMCALite` only on the P3 branch before Detect.

No P2 head, BiFPN, Powerful-IoU, NWD, loss change, dataset change, balanced sampling, full training, or test evaluation was used.

## Model Structure

EMCA-Lite keeps the standard YOLOv8n P3/P4/P5 Detect head. It adds a single `DentalEMCALite` block at the P3 feature map before Detect.

Build summary:

- Unfused build: `3,158,325` parameters, `8.9` GFLOPs
- Fused val summary: `3,007,358` parameters, `32.411` GFLOPs at `imgsz=1280`
- Baseline fused val summary: `3,006,233` parameters, `32.349` GFLOPs at `imgsz=1280`

The parameter and compute increase is small.

## Checks

- Local py_compile: passed
- Local model build: passed
- Server py_compile: passed
- Server model build: passed
- 1 epoch sanity: passed

Sanity run:

`runs/detect/dentex_v4_region_emca_lite_sanity/dentex_v4_region_yolov8n_emca_lite_640_sanity_20260523_190121_3405a79`

## Quick Run

- Run: `dentex_v4_region_yolov8n_emca_lite_1280_quick_b16_20260523_190225_3405a79`
- Run path: `runs/detect/dentex_v4_region_emca_lite_quick/dentex_v4_region_yolov8n_emca_lite_1280_quick_b16_20260523_190225_3405a79`
- Best weights: `runs/detect/dentex_v4_region_emca_lite_quick/dentex_v4_region_yolov8n_emca_lite_1280_quick_b16_20260523_190225_3405a79/weights/best.pt`
- Completed epochs: `50`
- Best epoch by training `results.csv`: `29`
- GPU peak from training log: about `8.14 GB`
- Best epoch metrics from `results.csv`: P `0.5829`, R `0.6485`, mAP50 `0.6203`, mAP50-95 `0.3840`
- Last epoch metrics from `results.csv`: P `0.5554`, R `0.6543`, mAP50 `0.5648`, mAP50-95 `0.3343`

## Unified Val Compare

Comparison output:

`reports/ablation/dentex_v4_region_emca_lite_quick_compare_20260523_192405/compare_20260523_192409_3405a79`

Both models were validated on the same v4 val split with `imgsz=1280`, `conf=0.001`, and `iou=0.7`.

| Model | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n 1280 quick b16 | 0.5691 | 0.6615 | 0.6479 | 0.3859 |
| YOLOv8n EMCA-Lite 1280 quick b16 | 0.5756 | 0.6505 | 0.6219 | 0.3880 |

## Per-Class Results

| Model | Class | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLOv8n 1280 quick b16 | Caries | 0.5116 | 0.5000 | 0.5155 | 0.3428 |
| YOLOv8n 1280 quick b16 | Periapical_Lesion | 0.6199 | 0.5812 | 0.6095 | 0.2523 |
| YOLOv8n 1280 quick b16 | Impacted | 0.5759 | 0.9032 | 0.8186 | 0.5627 |
| YOLOv8n EMCA-Lite 1280 quick b16 | Caries | 0.4366 | 0.5424 | 0.5217 | 0.3631 |
| YOLOv8n EMCA-Lite 1280 quick b16 | Periapical_Lesion | 0.6357 | 0.5529 | 0.5860 | 0.2446 |
| YOLOv8n EMCA-Lite 1280 quick b16 | Impacted | 0.6546 | 0.8561 | 0.7580 | 0.5561 |

## Delta Versus 1280 Quick Baseline

- Overall P: `+0.0065`
- Overall R: `-0.0110`
- Overall mAP50: `-0.0259`
- Overall mAP50-95: `+0.0020`
- Caries recall: `+0.0424`
- Caries mAP50-95: `+0.0203`
- Periapical_Lesion precision: `+0.0159`
- Periapical_Lesion recall: `-0.0282`
- Periapical_Lesion mAP50: `-0.0235`
- Periapical_Lesion mAP50-95: `-0.0077`
- Impacted recall: `-0.0471`
- Impacted mAP50: `-0.0605`
- Impacted mAP50-95: `-0.0066`

## Context Versus Prior Attempts

- P2 quick: rejected because Periapical_Lesion recall and Impacted recall dropped while compute increased.
- 1536 quick: improved overall mAP50-95 slightly, but Periapical_Lesion recall dropped and class 1 mAP50-95 was essentially unchanged.
- NWD-CIoU quick: overall mAP50-95 changed by only about `+0.0009`; Periapical_Lesion recall and mAP50 dropped.
- EMA-P3Lite quick: overall mAP50-95 dropped and Periapical_Lesion recall/mAP50 dropped.
- EMCA-Lite quick: overall mAP50-95 rises slightly, but the target Periapical_Lesion metrics regress.

## Recommendation

Do not run EMCA-Lite full training.

EMCA-Lite does not meet the success criteria:

- Periapical_Lesion recall drops from `0.5812` to `0.5529`.
- Periapical_Lesion mAP50 drops from `0.6095` to `0.5860`.
- Periapical_Lesion mAP50-95 drops from `0.2523` to `0.2446`.
- Overall mAP50 drops from `0.6479` to `0.6219`.
- Impacted degrades, especially mAP50: `0.8186 -> 0.7580`.
- Overall mAP50-95 improves only `+0.0020`, which is not enough to offset class 1 and Impacted regression.

The current main candidate should remain:

`v4_region YOLOv8n 1280 full best.pt`

No test split was used. Do not merge this experimental branch into `feat/dental-neckplus-v1` unless there is a later reason to preserve EMCA-Lite as an optional research feature.
