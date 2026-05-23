# DENTEX v4 Region 1536 Quick Ablation Report

## Summary

This report records the YOLOv8n 1536-input quick ablation on the DENTEX v4 region dataset.

- Dataset: `ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml`
- Run: `dentex_v4_region_yolov8n_1536_quick_b8_20260523_164759_6c6e5ed`
- Weights: `yolov8n.pt`
- Image size: `1536`
- Epochs: `50`
- Batch: `8`
- Workers: `24`
- BBox loss type: `ciou`
- Test split: not used

The run completed all 50 epochs with no OOM, NaN, path error, class error, or channel mismatch. The batch stayed at 8 and was not reduced.

## Training Status

- Run path: `runs/detect/dentex_v4_region_1536_quick/dentex_v4_region_yolov8n_1536_quick_b8_20260523_164759_6c6e5ed`
- Best weights: `runs/detect/dentex_v4_region_1536_quick/dentex_v4_region_yolov8n_1536_quick_b8_20260523_164759_6c6e5ed/weights/best.pt`
- Last weights: `runs/detect/dentex_v4_region_1536_quick/dentex_v4_region_yolov8n_1536_quick_b8_20260523_164759_6c6e5ed/weights/last.pt`
- Best epoch by training `results.csv`: `23`
- GPU peak from training log: about `5.59 GB`
- Final epoch metrics from `results.csv`: P `0.6034`, R `0.6578`, mAP50 `0.6035`, mAP50-95 `0.3530`
- Best epoch metrics from `results.csv`: P `0.5862`, R `0.6827`, mAP50 `0.6500`, mAP50-95 `0.3899`

## Unified Val Compare

Comparison output:

`reports/ablation/dentex_v4_region_1536_quick_compare_20260523_171657/compare_20260523_171700_6c6e5ed`

Both models were validated on the same v4 val split with `imgsz=1536`, `conf=0.001`, and `iou=0.7`. Note that the training input sizes are different: the baseline quick was trained at 1280, while this ablation was trained at 1536.

| Model | Val imgsz | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n 1280 quick b16 | 1536 | 0.5195 | 0.7051 | 0.6334 | 0.3774 |
| YOLOv8n 1536 quick b8 | 1536 | 0.5866 | 0.6830 | 0.6507 | 0.3928 |

## Per-Class Results

| Model | Class | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLOv8n 1280 quick b16 | Caries | 0.4716 | 0.5678 | 0.5237 | 0.3399 |
| YOLOv8n 1280 quick b16 | Periapical_Lesion | 0.5446 | 0.6444 | 0.5945 | 0.2634 |
| YOLOv8n 1280 quick b16 | Impacted | 0.5424 | 0.9032 | 0.7821 | 0.5289 |
| YOLOv8n 1536 quick b8 | Caries | 0.4930 | 0.5593 | 0.5310 | 0.3515 |
| YOLOv8n 1536 quick b8 | Periapical_Lesion | 0.6232 | 0.5576 | 0.6053 | 0.2630 |
| YOLOv8n 1536 quick b8 | Impacted | 0.6436 | 0.9321 | 0.8159 | 0.5640 |

## Delta Versus 1280 Quick

Unified validation at `imgsz=1536`:

- Overall P: `+0.0671`
- Overall R: `-0.0221`
- Overall mAP50: `+0.0173`
- Overall mAP50-95: `+0.0154`
- Caries mAP50-95: `+0.0117`, recall `-0.0085`
- Periapical_Lesion recall: `-0.0867`
- Periapical_Lesion mAP50: `+0.0109`
- Periapical_Lesion mAP50-95: `-0.0004`
- Impacted recall: `+0.0288`
- Impacted mAP50-95: `+0.0350`

## P2 Quick Context

The earlier YOLOv8n-P2 quick ablation was filtered out because it did not provide a clear benefit for this dataset:

- Overall mAP50-95 improved by only about `+0.0008` versus the 1280 quick baseline.
- Periapical_Lesion recall dropped from `0.5812` to `0.4894` in the original quick comparison.
- Impacted recall also degraded.
- P2 increased computation at 1280 without a clear gain.

The 1536 ablation is better than P2 as an optimization direction, but it still does not solve the main weakness: Periapical_Lesion recall does not improve.

## Recommendation

Do not start 1536 full training yet.

Although the 1536 quick run improves overall mAP50 and mAP50-95 under unified 1536 validation, the key target class does not improve enough:

- Periapical_Lesion recall falls from `0.6444` to `0.5576` in the unified 1536 comparison.
- Periapical_Lesion mAP50 rises only slightly from `0.5945` to `0.6053`.
- Periapical_Lesion mAP50-95 is essentially unchanged: `0.2634` to `0.2630`.

The current main candidate should remain the v4 region YOLOv8n 1280 full model:

`runs/detect/dentex_v4_region_full/dentex_v4_region_yolov8n_1280_full_b16_20260523_143152_1cfc9f5/weights/best.pt`

If the next step is still focused on small Periapical_Lesion boxes, the more targeted direction is to design a source-level NWD-CIoU mixed bbox loss experiment before training. This is better aligned with the observed issue: mAP50 is workable, while stricter localization mAP50-95 remains low for small lesion boxes.

No test split was used in this ablation.
