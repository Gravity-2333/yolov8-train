# DENTEX v4 Region YOLOv8m Extended Val Comparison Report

## 1. Scope

- Branch: `exp/dentex-v4-c2f-faster-lite`
- Experiment type: extended YOLOv8m final-candidate comparison
- Dataset config: `ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml`
- Split used in this report: `val`
- Image size: 1280
- Loss: CIoU
- YOLOv8m baseline batch: 8
- YOLOv8m C2f-Faster-lite batch: 8

This is an additional YOLOv8m experiment added after the previous YOLOv8n locked test. The current stage uses only val metrics for model comparison. No model was tuned or changed based on test results.

## 2. Training Status

| Model | Status | Epochs Logged | Best Epoch | Early Stop | GPU Peak |
| --- | --- | ---: | ---: | --- | ---: |
| YOLOv8m 1280 full baseline | completed | 137 | 37 | yes | 12.7 GB |
| YOLOv8m C2f-Faster-lite 1280 full | completed | 144 | 44 | yes | 12.4 GB |

Best weights:

- YOLOv8m baseline: `/root/autodl-tmp/work/yolov8-train/runs/detect/dentex_v4_region_yolov8m_full/dentex_v4_region_yolov8m_1280_full_b8_20260524_104521_1fdf7df/weights/best.pt`
- YOLOv8m C2f-Faster-lite: `/root/autodl-tmp/work/yolov8-train/runs/detect/dentex_v4_region_yolov8m_c2f_faster_lite_full/dentex_v4_region_yolov8m_c2f_faster_lite_1280_full_b8_20260524_132149_1fdf7df/weights/best.pt`

## 3. Val Overall Comparison

| Model | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n 1280 full | 0.5984 | 0.6594 | 0.6652 | 0.4065 |
| YOLOv8n C2f-Faster-lite 1280 full | 0.6194 | 0.6021 | 0.6593 | 0.3918 |
| YOLOv8m 1280 full | 0.5721 | 0.6879 | 0.6602 | 0.3822 |
| YOLOv8m C2f-Faster-lite 1280 full | 0.6073 | 0.6923 | 0.6555 | 0.3929 |

On val, YOLOv8m does not outperform the YOLOv8n baseline in mAP50-95. YOLOv8m baseline improves recall compared with YOLOv8n baseline, but has lower precision and lower mAP50-95.

## 4. Val Per-Class Comparison

| Model | Class | P | R | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLOv8n 1280 full | Caries | 0.4857 | 0.5678 | 0.5799 | 0.3653 |
| YOLOv8n 1280 full | Periapical_Lesion | 0.6145 | 0.5718 | 0.5943 | 0.2573 |
| YOLOv8n 1280 full | Impacted | 0.6952 | 0.8387 | 0.8215 | 0.5971 |
| YOLOv8n C2f-Faster-lite 1280 full | Caries | 0.5242 | 0.5000 | 0.5725 | 0.3719 |
| YOLOv8n C2f-Faster-lite 1280 full | Periapical_Lesion | 0.6990 | 0.4353 | 0.6001 | 0.2482 |
| YOLOv8n C2f-Faster-lite 1280 full | Impacted | 0.6351 | 0.8710 | 0.8053 | 0.5553 |
| YOLOv8m 1280 full | Caries | 0.4554 | 0.6186 | 0.5939 | 0.3800 |
| YOLOv8m 1280 full | Periapical_Lesion | 0.6157 | 0.6709 | 0.6334 | 0.2680 |
| YOLOv8m 1280 full | Impacted | 0.6451 | 0.7742 | 0.7534 | 0.4986 |
| YOLOv8m C2f-Faster-lite 1280 full | Caries | 0.5058 | 0.6271 | 0.5962 | 0.3989 |
| YOLOv8m C2f-Faster-lite 1280 full | Periapical_Lesion | 0.6598 | 0.5788 | 0.6131 | 0.2569 |
| YOLOv8m C2f-Faster-lite 1280 full | Impacted | 0.6563 | 0.8710 | 0.7574 | 0.5229 |

## 5. Model Cost Comparison

| Model | Params | GFLOPs | Size | Inference |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n 1280 full | 3.006M | 32.349 | 6.03 MB | 1.46 ms/img |
| YOLOv8n C2f-Faster-lite 1280 full | 2.602M | 29.764 | 5.26 MB | 1.06 ms/img |
| YOLOv8m 1280 full | 25.841M | 314.780 | 49.69 MB | 5.79 ms/img |
| YOLOv8m C2f-Faster-lite 1280 full | 20.690M | 275.030 | 39.86 MB | 5.42 ms/img |

On YOLOv8m, C2f-Faster-lite still provides a clear lightweight benefit:

- Params: 25.841M -> 20.690M, down 19.9%
- GFLOPs: 314.780 -> 275.030, down 12.6%
- Model size: 49.69 MB -> 39.86 MB, down 19.8%
- Inference time: 5.79 ms/img -> 5.42 ms/img, down 6.5%

## 6. Val Decision

- YOLOv8m baseline is not better than YOLOv8n baseline on val mAP50-95.
- YOLOv8m C2f-Faster-lite is lighter than YOLOv8m baseline and slightly better on val mAP50-95, but still below YOLOv8n baseline.
- C2f-Faster-lite remains effective as a lightweight structure on YOLOv8m.
- The four models are locked for extended final test comparison:
  - YOLOv8n 1280 full
  - YOLOv8n C2f-Faster-lite 1280 full
  - YOLOv8m 1280 full
  - YOLOv8m C2f-Faster-lite 1280 full

The next step is a single extended final test comparison across these four locked candidates. Test results will be used only for final verification, not for further tuning, retraining, or model changes.
