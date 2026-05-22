# DENTEX v2 Main Baseline Report

## Dataset

- dataset: `dentex_yolov8_3cls_v2`
- server path: `/root/autodl-tmp/work/datasets_yolo8/dentex_yolov8_3cls_v2`
- server yaml: `ultralytics/cfg/datasets/dentex_yolov8_3cls_v2_server.yaml`

## Dataset Size

| split | images | boxes |
|---|---:|---:|
| train | 705 | 3529 |
| val | 50 | 182 |
| test | 250 | 1043 |

## Class Instances

| split | Caries | Periapical Lesion | Impacted |
|---|---:|---:|---:|
| train | 2767 | 158 | 604 |
| val | 133 | 9 | 40 |
| test | 747 | 75 | 221 |

## Quick Baseline

- model: `yolov8n.pt`
- imgsz: `960`
- batch: `32`
- workers: `24`
- epochs: `50`
- best epoch: `25`

### Overall Metrics

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.6425 | 0.5683 | 0.5861 | 0.3906 |

### Per-Class Metrics

| class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Caries | 0.5675 | 0.5624 | 0.5790 | 0.3850 |
| Periapical Lesion | 0.5774 | 0.3333 | 0.3536 | 0.2296 |
| Impacted | 0.7824 | 0.8093 | 0.8259 | 0.5570 |

## Conclusion

DENTEX v2 is the main dataset for the next stage. Compared with the previous v1
diagnostics, it gives a clearer training signal for Periapical Lesion, but the
dataset is small and the validation split is especially small. Single-run val
metrics should be treated cautiously, and final model selection should avoid
repeated tuning on the test split.
