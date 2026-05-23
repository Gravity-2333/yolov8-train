# DENTEX v4 Region C2f-Faster-lite Full Report

## Summary

This report records the full training and validation comparison for YOLOv8n + C2f-Faster-lite on DENTEX v4_region.

- Branch: `exp/dentex-v4-c2f-faster-lite`
- Implementation commit: `42a8c170b Implement DENTEX v4 C2f-Faster-lite model`
- Quick report commit: `87db6ce Add DENTEX v4 C2f-Faster-lite quick ablation`
- Dataset: `ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml`
- Model YAML: `ultralytics/cfg/models/v8/yolov8n_dental_c2f_faster_lite.yaml`
- Train image size: `1280`
- Epochs requested: `200`
- Batch: `16`
- Workers: `24`
- Loss: original CIoU
- Test split: not used

The full run completed without OOM, NaN, channel mismatch, Detect shape error, path error, or class error. It stopped before 200 epochs after early stopping behavior.

## Runs

Baseline full:

`runs/detect/dentex_v4_region_full/dentex_v4_region_yolov8n_1280_full_b16_20260523_143152_1cfc9f5`

C2f-Faster-lite full:

`runs/detect/dentex_v4_region_c2f_faster_lite_full/dentex_v4_region_yolov8n_c2f_faster_lite_1280_full_b16_20260523_204914_87db6ce`

C2f-Faster-lite best weights:

`runs/detect/dentex_v4_region_c2f_faster_lite_full/dentex_v4_region_yolov8n_c2f_faster_lite_1280_full_b16_20260523_204914_87db6ce/weights/best.pt`

## Training Outcome

From `results.csv` of the C2f-Faster-lite full run:

- Completed rows: `128`
- Last epoch: `128`
- Best epoch by `mAP50-95`: `28`
- Best training-log metrics:
  - Precision: `0.6127`
  - Recall: `0.6052`
  - mAP50: `0.6596`
  - mAP50-95: `0.3875`
- Last row metrics:
  - Precision: `0.6226`
  - Recall: `0.5867`
  - mAP50: `0.5802`
  - mAP50-95: `0.3423`

## Unified Val Compare

Comparison output:

`reports/ablation/dentex_v4_region_c2f_faster_lite_full_compare_20260523_224849/compare_20260523_224852_87db6ce`

Both models were validated on the same DENTEX v4_region val split with:

- `imgsz=1280`
- `conf=0.001`
- `iou=0.7`
- `device=0`

### Overall Metrics

| Model | Precision | Recall | mAP50 | mAP50-95 | Params | GFLOPs | Size MB | Inference ms/img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n full baseline | 0.5984 | 0.6594 | 0.6652 | 0.4065 | 3,006,233 | 32.349 | 6.03 | 1.56 |
| C2f-Faster-lite full | 0.6194 | 0.6021 | 0.6593 | 0.3918 | 2,602,073 | 29.764 | 5.26 | 1.05 |

### Relative Change

C2f-Faster-lite compared with YOLOv8n full baseline:

- Precision: `+0.0210`
- Recall: `-0.0573`
- mAP50: `-0.0059`
- mAP50-95: `-0.0148`
- Params: about `-13.4%`
- GFLOPs: about `-8.0%`
- Model size: about `-12.8%`
- Inference time: about `-32.7%` in this validation run

## Per-class Metrics

| Model | Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLOv8n full baseline | Caries | 0.4857 | 0.5678 | 0.5799 | 0.3653 |
| YOLOv8n full baseline | Periapical_Lesion | 0.6145 | 0.5718 | 0.5943 | 0.2573 |
| YOLOv8n full baseline | Impacted | 0.6952 | 0.8387 | 0.8215 | 0.5971 |
| C2f-Faster-lite full | Caries | 0.5242 | 0.5000 | 0.5725 | 0.3719 |
| C2f-Faster-lite full | Periapical_Lesion | 0.6990 | 0.4353 | 0.6001 | 0.2482 |
| C2f-Faster-lite full | Impacted | 0.6351 | 0.8710 | 0.8053 | 0.5553 |

### Per-class Interpretation

- `Caries`: C2f-Faster-lite improves precision and mAP50-95, but recall drops.
- `Periapical_Lesion`: precision and mAP50 improve slightly, but recall drops heavily from `0.5718` to `0.4353`; mAP50-95 also drops slightly.
- `Impacted`: recall improves, but precision, mAP50, and mAP50-95 drop.

## Conclusion

C2f-Faster-lite is a valid lightweight model optimization result, but it is not the best accuracy candidate.

It achieves meaningful compression and faster inference:

- `-13.4%` params
- `-8.0%` GFLOPs
- `-12.8%` model size
- lower measured inference latency on val

However, the full-training accuracy comparison does not support replacing the YOLOv8n 1280 full baseline as the main final model:

- overall mAP50-95 drops from `0.4065` to `0.3918`;
- overall recall drops from `0.6594` to `0.6021`;
- Periapical_Lesion recall drops from `0.5718` to `0.4353`.

Recommended current position:

1. Keep `v4_region YOLOv8n 1280 full best.pt` as the main accuracy candidate.
2. Keep `C2f-Faster-lite full best.pt` as a lightweight optimization candidate for the project write-up.
3. Do not merge `exp/dentex-v4-c2f-faster-lite` back to the stable branch unless the project explicitly wants the lightweight variant in the main code path.
4. Do not run final test yet until the final model-selection rule is confirmed.

No test evaluation was run in this stage.
