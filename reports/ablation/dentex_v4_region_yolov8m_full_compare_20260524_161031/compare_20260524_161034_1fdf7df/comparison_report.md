# Dental Ablation Comparison

- Split: `val`

| experiment | P | R | mAP50 | mAP50-95 | GFLOPs | size MB |
|---|---:|---:|---:|---:|---:|---:|
| yolov8n_1280_full | 0.5984 | 0.6594 | 0.6652 | 0.4065 | 32.349 | 6.03 |
| yolov8n_c2f_faster_lite_1280_full | 0.6194 | 0.6021 | 0.6593 | 0.3918 | 29.764 | 5.26 |
| yolov8m_1280_full | 0.5721 | 0.6879 | 0.6602 | 0.3822 | 314.780 | 49.69 |
| yolov8m_c2f_faster_lite_1280_full | 0.6073 | 0.6923 | 0.6555 | 0.3929 | 275.030 | 39.86 |

## Notes
- Results use the `val` split.
- Test split should be used only for locked final confirmation.
- First-stage small/medium/large analysis should be read together with `dataset_audit_dental.py` outputs.
