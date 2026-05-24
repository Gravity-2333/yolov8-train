# Dental Ablation Comparison

| experiment | P | R | mAP50 | mAP50-95 | GFLOPs | size MB |
|---|---:|---:|---:|---:|---:|---:|
| dentex_v4_region_yolov8n_1280_full_b16 | 0.5984 | 0.6594 | 0.6652 | 0.4065 | 32.349 | 6.03 |
| dentex_v4_region_yolov8n_c2f_faster_lite_1280_full_b16 | 0.6194 | 0.6021 | 0.6593 | 0.3918 | 29.764 | 5.26 |

## Notes
- Validation uses the val split only; reserve test for final confirmation.
- First-stage small/medium/large analysis should be read together with `dataset_audit_dental.py` outputs.
