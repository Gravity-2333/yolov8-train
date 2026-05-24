# DENTEX v4 Region Extended Final Test Report

## 1. Scope

- Branch: `exp/dentex-v4-c2f-faster-lite`
- Comparison type: extended final test comparison
- Dataset config: `ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml`
- Split: `test`
- Test images: 749
- Test instances: 1582
- Background images: 11
- Corrupt images: 0
- Image size: 1280
- Confidence threshold: 0.001
- IoU threshold: 0.7

This test was used only for extended final verification after the additional YOLOv8m experiments were trained and locked. No model was tuned, retrained, or changed based on test results. All training and optimization stop after this comparison.

## 2. Locked Test Candidates

| Model | Role |
| --- | --- |
| YOLOv8n 1280 full | previous stable baseline |
| YOLOv8n C2f-Faster-lite 1280 full | lightweight YOLOv8n optimization |
| YOLOv8m 1280 full | larger model for higher precision |
| YOLOv8m C2f-Faster-lite 1280 full | larger C2f-Faster-lite variant |

The YOLOv8m experiments were added after the previous YOLOv8n locked test stage. They are part of this extended final comparison, not part of the earlier timeline.

## 3. Test Overall Comparison

| Model | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n 1280 full | 0.6340 | 0.6828 | 0.6488 | 0.3729 |
| YOLOv8n C2f-Faster-lite 1280 full | 0.5885 | 0.6793 | 0.6483 | 0.3748 |
| YOLOv8m 1280 full | 0.6437 | 0.7175 | 0.6660 | 0.3692 |
| YOLOv8m C2f-Faster-lite 1280 full | 0.6253 | 0.6923 | 0.6491 | 0.3804 |

Key observations:

- Highest mAP50-95: YOLOv8m C2f-Faster-lite, 0.3804.
- Highest mAP50: YOLOv8m baseline, 0.6660.
- Highest recall: YOLOv8m baseline, 0.7175.
- YOLOv8m baseline improves mAP50 over YOLOv8n baseline by +0.0171, but has lower mAP50-95.
- YOLOv8m C2f-Faster-lite improves mAP50-95 over YOLOv8m baseline by +0.0112 while reducing model cost.

## 4. Test Per-Class Comparison

| Model | Class | P | R | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLOv8n 1280 full | Caries | 0.4172 | 0.5045 | 0.4140 | 0.2845 |
| YOLOv8n 1280 full | Periapical_Lesion | 0.5969 | 0.6168 | 0.6134 | 0.2619 |
| YOLOv8n 1280 full | Impacted | 0.8879 | 0.9272 | 0.9192 | 0.5723 |
| YOLOv8n C2f-Faster-lite 1280 full | Caries | 0.3926 | 0.5546 | 0.4263 | 0.2861 |
| YOLOv8n C2f-Faster-lite 1280 full | Periapical_Lesion | 0.6229 | 0.5365 | 0.6023 | 0.2581 |
| YOLOv8n C2f-Faster-lite 1280 full | Impacted | 0.7499 | 0.9468 | 0.9162 | 0.5803 |
| YOLOv8m 1280 full | Caries | 0.4429 | 0.6154 | 0.4522 | 0.2950 |
| YOLOv8m 1280 full | Periapical_Lesion | 0.6003 | 0.6515 | 0.6377 | 0.2633 |
| YOLOv8m 1280 full | Impacted | 0.8880 | 0.8855 | 0.9080 | 0.5492 |
| YOLOv8m C2f-Faster-lite 1280 full | Caries | 0.4136 | 0.5832 | 0.4581 | 0.3142 |
| YOLOv8m C2f-Faster-lite 1280 full | Periapical_Lesion | 0.6305 | 0.5743 | 0.5926 | 0.2447 |
| YOLOv8m C2f-Faster-lite 1280 full | Impacted | 0.8316 | 0.9193 | 0.8966 | 0.5823 |

## 5. Cost Comparison

| Model | Params | GFLOPs | Size | Inference |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n 1280 full | 3.006M | 32.349 | 6.03 MB | 1.34 ms/img |
| YOLOv8n C2f-Faster-lite 1280 full | 2.602M | 29.764 | 5.26 MB | 1.02 ms/img |
| YOLOv8m 1280 full | 25.841M | 314.780 | 49.69 MB | 5.66 ms/img |
| YOLOv8m C2f-Faster-lite 1280 full | 20.690M | 275.030 | 39.86 MB | 5.34 ms/img |

YOLOv8m C2f-Faster-lite compared with YOLOv8m baseline:

- Params: 25.841M -> 20.690M, down 19.9%.
- GFLOPs: 314.780 -> 275.030, down 12.6%.
- Model size: 49.69 MB -> 39.86 MB, down 19.8%.
- Inference time: 5.66 ms/img -> 5.34 ms/img, down 5.6%.

YOLOv8n C2f-Faster-lite remains the strongest lightweight deployment candidate because it is much smaller and faster than both YOLOv8m variants while preserving competitive test mAP50-95.

## 6. Final Decision

- Final highest mAP50-95 model: YOLOv8m C2f-Faster-lite 1280 full, test mAP50-95 0.3804.
- Final highest mAP50 model: YOLOv8m 1280 full, test mAP50 0.6660.
- Final lightweight optimization model: YOLOv8n C2f-Faster-lite 1280 full.
- YOLOv8m is useful as an extended high-precision candidate because it improves test mAP50 and recall, but the mAP50-95 gain is only realized by the YOLOv8m C2f-Faster-lite variant.
- C2f-Faster-lite remains effective on YOLOv8m because it reduces params, GFLOPs, size, and inference time, and improves mAP50-95 versus YOLOv8m baseline.

## 7. Locked Test Notes

- Test was used only for extended final verification.
- No test-driven tuning was performed.
- No additional model optimization was started after this test.
- No further training was started after this test.
- The experiment branch was not merged.
