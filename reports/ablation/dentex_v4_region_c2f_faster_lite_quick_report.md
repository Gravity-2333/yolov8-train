# DENTEX v4 Region C2f-Faster-lite Quick Ablation Report

## Summary

This report records the YOLOv8n + C2f-Faster-lite quick ablation on DENTEX v4_region.

- Experiment branch: `exp/dentex-v4-c2f-faster-lite`
- Implementation commit: `42a8c170b Implement DENTEX v4 C2f-Faster-lite model`
- Dataset: `ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml`
- Model YAML: `ultralytics/cfg/models/v8/yolov8n_dental_c2f_faster_lite.yaml`
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
  - Added `FasterLiteBottleneck`.
  - Added `C2fFasterLite`.
- `ultralytics/nn/modules/__init__.py`
  - Exported `C2fFasterLite`.
- `ultralytics/nn/tasks.py`
  - Registered `C2fFasterLite` as a C2f-style base and repeat module.
- `ultralytics/cfg/models/v8/yolov8n_dental_c2f_faster_lite.yaml`
  - Added a YOLOv8n variant that replaces only two backbone C2f layers.

No P2 head, attention module, BiFPN, NWD, loss change, dataset change, balanced sampling, full training, or test evaluation was used.

## Replaced Layers

The replacement is conservative and only targets the middle/deep backbone:

- Layer 6, P4 backbone: `C2f -> C2fFasterLite`
- Layer 8, P5 backbone: `C2f -> C2fFasterLite`

The neck remains the original YOLOv8n neck. Detect remains P3/P4/P5.

## Model Size And Compute

Unfused build:

| Model | Params | GFLOPs |
|---|---:|---:|
| YOLOv8n baseline | 3,157,200 | 8.9 |
| YOLOv8n C2f-Faster-lite | 2,752,848 | 8.2 |

Unified val fused summary at `imgsz=1280`:

| Model | Params | GFLOPs | size MB |
|---|---:|---:|---:|
| YOLOv8n baseline | 3,006,233 | 32.349 | 6.02 |
| YOLOv8n C2f-Faster-lite | 2,602,073 | 29.764 | 5.25 |

Relative reduction from fused val summary:

- Params: about `-13.4%`
- GFLOPs: about `-8.0%`
- Model size: about `-12.8%`

This satisfies the lightweight optimization target.

## Checks

- Local py_compile: passed
- Local model build: passed
- Server py_compile: passed
- Server model build: passed
- 1 epoch sanity: passed

Sanity run:

`runs/detect/dentex_v4_region_c2f_faster_lite_sanity/dentex_v4_region_yolov8n_c2f_faster_lite_640_sanity_20260523_194947_42a8c17`

## Quick Run

- Run: `dentex_v4_region_yolov8n_c2f_faster_lite_1280_quick_b16_20260523_195100_42a8c17`
- Run path: `runs/detect/dentex_v4_region_c2f_faster_lite_quick/dentex_v4_region_yolov8n_c2f_faster_lite_1280_quick_b16_20260523_195100_42a8c17`
- Best weights: `runs/detect/dentex_v4_region_c2f_faster_lite_quick/dentex_v4_region_yolov8n_c2f_faster_lite_1280_quick_b16_20260523_195100_42a8c17/weights/best.pt`
- Completed epochs: `50`
- Best epoch by training `results.csv`: `25`
- GPU peak from training log: about `7.92 GB`
- Best epoch metrics from `results.csv`: P `0.5906`, R `0.6593`, mAP50 `0.6440`, mAP50-95 `0.3938`
- Last epoch metrics from `results.csv`: P `0.5889`, R `0.6711`, mAP50 `0.6255`, mAP50-95 `0.3686`

## Unified Val Compare

Comparison output:

`reports/ablation/dentex_v4_region_c2f_faster_lite_quick_compare_20260523_201921/compare_20260523_201925_42a8c17`

Both models were validated on the same v4 val split with `imgsz=1280`, `conf=0.001`, and `iou=0.7`.

| Model | P | R | mAP50 | mAP50-95 | Params | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n 1280 quick b16 | 0.5691 | 0.6615 | 0.6479 | 0.3859 | 3,006,233 | 32.349 |
| C2f-Faster-lite 1280 quick b16 | 0.5928 | 0.6593 | 0.6420 | 0.3916 | 2,602,073 | 29.764 |

## Per-Class Results

| Model | Class | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLOv8n 1280 quick b16 | Caries | 0.5116 | 0.5000 | 0.5155 | 0.3428 |
| YOLOv8n 1280 quick b16 | Periapical_Lesion | 0.6199 | 0.5812 | 0.6095 | 0.2523 |
| YOLOv8n 1280 quick b16 | Impacted | 0.5759 | 0.9032 | 0.8186 | 0.5627 |
| C2f-Faster-lite 1280 quick b16 | Caries | 0.4828 | 0.6271 | 0.5568 | 0.3715 |
| C2f-Faster-lite 1280 quick b16 | Periapical_Lesion | 0.6725 | 0.5765 | 0.6065 | 0.2571 |
| C2f-Faster-lite 1280 quick b16 | Impacted | 0.6232 | 0.7742 | 0.7626 | 0.5461 |

## Delta Versus 1280 Quick Baseline

- Overall P: `+0.0237`
- Overall R: `-0.0022`
- Overall mAP50: `-0.0059`
- Overall mAP50-95: `+0.0056`
- Params: about `-13.4%`
- GFLOPs: about `-8.0%`
- Caries recall: `+0.1271`
- Caries mAP50-95: `+0.0287`
- Periapical_Lesion precision: `+0.0527`
- Periapical_Lesion recall: `-0.0047`
- Periapical_Lesion mAP50: `-0.0030`
- Periapical_Lesion mAP50-95: `+0.0048`
- Impacted recall: `-0.1290`
- Impacted mAP50: `-0.0560`
- Impacted mAP50-95: `-0.0166`

## Context Versus Prior Attempts

- P2 quick: rejected because Periapical_Lesion recall and Impacted recall dropped while compute increased.
- 1536 quick: improved overall mAP50-95 slightly, but Periapical_Lesion recall dropped and class 1 mAP50-95 was essentially unchanged.
- NWD-CIoU quick: overall mAP50-95 changed by only about `+0.0009`; Periapical_Lesion recall and mAP50 dropped.
- EMA-P3Lite quick: overall mAP50-95 dropped and Periapical_Lesion recall/mAP50 dropped.
- EMCA-Lite quick: overall mAP50-95 rose only `+0.0020`, but Periapical_Lesion and Impacted regressed.
- C2f-Faster-lite quick: achieves a clearer lightweight model tradeoff: fewer params, lower GFLOPs, slightly higher overall mAP50-95, and almost unchanged Periapical_Lesion recall.

## Recommendation

C2f-Faster-lite is the first optimization that can reasonably be presented as a model optimization result, because it reduces compute and parameters while preserving or slightly improving overall mAP50-95.

However, the Impacted class regresses:

- Impacted recall: `0.9032 -> 0.7742`
- Impacted mAP50: `0.8186 -> 0.7626`
- Impacted mAP50-95: `0.5627 -> 0.5461`

Recommendation:

- It is reasonable to consider one C2f-Faster-lite full run if the project needs a lightweight optimized model artifact.
- Do not treat it as strictly better than the current full baseline for all classes.
- Keep `v4_region YOLOv8n 1280 full best.pt` as the current main accuracy candidate until a C2f-Faster-lite full run is evaluated on val.
- Do not run test yet.

No test split was used in this experiment. Do not merge this branch into `feat/dental-neckplus-v1` until the user confirms whether C2f-Faster-lite should enter full training.
