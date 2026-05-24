# DENTEX v4 Region Locked Final Test Report

## 1. Run Scope

- Branch: `exp/dentex-v4-c2f-faster-lite`
- Final test compare output: `reports/final_test/dentex_v4_region_locked_final_test_20260524_102206/compare_20260524_102209_1fdf7df`
- Dataset config: `ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml`
- Split: `test`
- Test images: 749
- Test instances: 1582
- Background images: 11
- Corrupt images: 0
- Image size: 1280
- Confidence threshold: 0.001
- IoU threshold: 0.7
- Device: 0

This final test was run once for locked model verification only. No new training, tuning, architecture changes, or post-test optimization were performed. The experiment branch was not merged.

## 2. Locked Candidate Models

| Role | Name | Weight |
| --- | --- | --- |
| Highest-precision final candidate | `yolov8n_1280_full_precision_final` | `/root/autodl-tmp/work/yolov8-train/runs/detect/dentex_v4_region_full/dentex_v4_region_yolov8n_1280_full_b16_20260523_143152_1cfc9f5/weights/best.pt` |
| Lightweight optimization final candidate | `c2f_faster_lite_1280_full_lightweight_final` | `/root/autodl-tmp/work/yolov8-train/runs/detect/dentex_v4_region_c2f_faster_lite_full/dentex_v4_region_yolov8n_c2f_faster_lite_1280_full_b16_20260523_204914_87db6ce/weights/best.pt` |

Only these two locked `best.pt` weights were compared on the test split.

## 3. Test Overall Metrics

| Model | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n 1280 full precision final | 0.6340 | 0.6828 | 0.6488 | 0.3729 |
| C2f-Faster-lite 1280 full lightweight final | 0.5885 | 0.6793 | 0.6483 | 0.3748 |
| Delta, C2f-Faster-lite minus baseline | -0.0455 | -0.0035 | -0.0006 | +0.0020 |

## 4. Baseline Full Test Per-Class Metrics

| Class | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Caries | 0.4172 | 0.5045 | 0.4140 | 0.2845 |
| Periapical_Lesion | 0.5969 | 0.6168 | 0.6134 | 0.2619 |
| Impacted | 0.8879 | 0.9272 | 0.9192 | 0.5723 |

## 5. C2f-Faster-lite Full Test Per-Class Metrics

| Class | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Caries | 0.3926 | 0.5546 | 0.4263 | 0.2861 |
| Periapical_Lesion | 0.6229 | 0.5365 | 0.6023 | 0.2581 |
| Impacted | 0.7499 | 0.9468 | 0.9162 | 0.5803 |

## 6. Lightweight Benefit

| Metric | YOLOv8n Full | C2f-Faster-lite Full | Change |
| --- | ---: | ---: | ---: |
| Params | 3.006M | 2.602M | -13.4% |
| GFLOPs | 32.349 | 29.764 | -8.0% |
| Model size | 6.03 MB | 5.26 MB | -12.8% |
| Inference time | 1.33 ms/img | 1.03 ms/img | -22.6% |

C2f-Faster-lite keeps test mAP50 almost unchanged and slightly improves test mAP50-95 while reducing parameters, GFLOPs, model size, and inference time. Precision decreases, and Periapical_Lesion recall decreases from 0.6168 to 0.5365, so it should not replace the baseline as the highest-precision model.

## 7. Full Val Background

| Model | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n full baseline val | 0.5984 | 0.6594 | 0.6652 | 0.4065 |
| C2f-Faster-lite full val | 0.6194 | 0.6021 | 0.6593 | 0.3918 |

The val comparison already showed that C2f-Faster-lite is not the strongest precision candidate, but has a clear lightweight advantage.

## 8. Final Recommendation

- Highest-precision final model: `v4_region YOLOv8n 1280 full best.pt`
- Lightweight optimization model: `v4_region YOLOv8n C2f-Faster-lite 1280 full best.pt`
- C2f-Faster-lite remains acceptable as a lightweight optimization result because test mAP50 is effectively tied, test mAP50-95 is slightly higher, and compute/model size are lower.
- C2f-Faster-lite is not recommended as the final highest-precision main model because precision and Periapical_Lesion recall decline.

## 9. Locked Test Notes

- Test was used only for final verification.
- No tuning or model selection loop was performed based on test results.
- No new training was started.
- No full rerun was started.
- No YOLOv8m, P2, 1536, NWD, EMA, EMCA, or BiFPN experiment was run.
- No experiment branch merge was performed.
