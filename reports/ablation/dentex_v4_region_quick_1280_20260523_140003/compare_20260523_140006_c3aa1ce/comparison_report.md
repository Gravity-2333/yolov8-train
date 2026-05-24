# Dental Ablation Comparison

| experiment | P | R | mAP50 | mAP50-95 | GFLOPs | size MB |
|---|---:|---:|---:|---:|---:|---:|
| dentex_v4_region_yolov8n_1280_quick_b16 | 0.5691 | 0.6615 | 0.6479 | 0.3859 | 32.349 | 6.02 |

## Notes
- Validation uses the val split only; reserve test for final confirmation.
- First-stage small/medium/large analysis should be read together with `dataset_audit_dental.py` outputs.
