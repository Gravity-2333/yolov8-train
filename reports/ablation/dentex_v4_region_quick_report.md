# DENTEX v4 Region Quick Report

## Dataset

- Dataset: `dentex_yolov8_3cls_v4_region`
- Server path: `/root/autodl-tmp/work/datasets_yolo8/dentex_yolov8_3cls_v4_region`
- Data config: `ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml`
- Task classes:
  - `0 Caries`
  - `1 Periapical_Lesion`
  - `2 Impacted`

## Audit Summary

The v4 region dataset audit passed on the server. No class id overflow, bad bbox, outside bbox, missing labels, or bad label format was found.

| Split | Images | Labels | Boxes | Empty Labels |
| --- | ---: | ---: | ---: | ---: |
| train | 2560 | 2560 | 5705 | 27 |
| val | 315 | 315 | 574 | 4 |
| test | 749 | 749 | 1582 | 11 |

Class distribution:

| Split | Caries boxes/images | Periapical_Lesion boxes/images | Impacted boxes/images |
| --- | ---: | ---: | ---: |
| train | 2179 / 546 | 3017 / 1971 | 509 / 210 |
| val | 118 / 38 | 425 / 272 | 31 / 13 |
| test | 559 / 174 | 835 / 554 | 188 / 74 |

Visual audit output was generated under `reports/visual_audit/dentex_v4_region_check/`. The sampled class 1 boxes are small periapical lesion region boxes, and no v2 tooth-level Periapical_Lesion leak was found in the sampled visual check.

## Quick Training

This was a quick validation run only. It did not use test set evaluation.

Training command profile:

- model: `yolov8n.pt`
- weights: `yolov8n.pt`
- imgsz: `1280`
- epochs: `50`
- batch: `16`
- workers: `24`
- bbox loss type: `ciou`
- cache: disabled
- plots: disabled by training script

Run directory:

`runs/detect/dentex_v4_region_quick/dentex_v4_region_yolov8n_1280_quick_b16_20260523_133553_c3aa1ce`

The run completed all 50 epochs successfully. No OOM, NaN, channel mismatch, path error, or class error was observed.

## Unified Val Compare

Compare output:

`reports/ablation/dentex_v4_region_quick_1280_20260523_140003/compare_20260523_140006_c3aa1ce`

Overall validation metrics on v4 val:

| Experiment | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| dentex_v4_region_yolov8n_1280_quick_b16 | 0.5691 | 0.6615 | 0.6479 | 0.3859 |

Per-class validation metrics:

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Caries | 0.5116 | 0.5000 | 0.5155 | 0.3428 |
| Periapical_Lesion | 0.6199 | 0.5812 | 0.6095 | 0.2523 |
| Impacted | 0.5759 | 0.9032 | 0.8186 | 0.5627 |

## Interpretation

The v4 region formulation is viable for class 1 region-level Periapical_Lesion detection. The quick run already gives class 1 recall around `0.58` and mAP50 around `0.61` on the v4 validation set, which suggests the small lesion-region boxes can be learned by YOLOv8n at `imgsz=1280`.

Caries is weaker than Impacted and Periapical_Lesion in this quick run. This is expected because v4 removes all v2 images containing old Periapical_Lesion boxes, which also discards some Caries and Impacted examples. A full run should determine whether Caries recovers with longer training.

Recommended next step: run v4 region full training with YOLOv8n before trying larger models or architecture changes. Keep using val only for model selection, and reserve test for final confirmation.
