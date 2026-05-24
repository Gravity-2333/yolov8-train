# Dental Ablation Comparison

- Split: `test`

| experiment | P | R | mAP50 | mAP50-95 | GFLOPs | size MB |
|---|---:|---:|---:|---:|---:|---:|
| yolov8n_1280_full | 0.6340 | 0.6828 | 0.6488 | 0.3729 | 32.349 | 6.03 |
| yolov8n_c2f_faster_lite_1280_full | 0.5885 | 0.6793 | 0.6483 | 0.3748 | 29.764 | 5.26 |
| yolov8m_1280_full | 0.6437 | 0.7175 | 0.6660 | 0.3692 | 314.780 | 49.69 |
| yolov8m_c2f_faster_lite_1280_full | 0.6253 | 0.6923 | 0.6491 | 0.3804 | 275.030 | 39.86 |

## Notes
- Results use the `test` split.
- Test split should be used only for locked final confirmation.
- First-stage small/medium/large analysis should be read together with `dataset_audit_dental.py` outputs.
