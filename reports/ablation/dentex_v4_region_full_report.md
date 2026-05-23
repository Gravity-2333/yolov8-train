# DENTEX v4 Region Full Report

## Run

- Run name: `dentex_v4_region_yolov8n_1280_full_b16_20260523_143152_1cfc9f5`
- Branch: `feat/dental-neckplus-v1`
- Commit: `1cfc9f5`
- Data: `ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml`
- Model: `yolov8n.pt`
- Image size: `1280`
- Batch: `16`
- Workers: `24`
- Epoch target: `200`
- BBox loss type: `ciou`
- Test set: not used

## Training Status

The full baseline completed successfully and stopped by EarlyStopping.

- Completed epochs: `135`
- Best epoch: `35`
- Early stop: yes, no improvement in the last `100` epochs
- GPU peak from log: about `8.01 GB`
- Errors observed: no OOM, no NaN, no channel mismatch, no path error, no class error

Best model path on server:

`runs/detect/dentex_v4_region_full/dentex_v4_region_yolov8n_1280_full_b16_20260523_143152_1cfc9f5/weights/best.pt`

## Unified Val Metrics

Validation was run with `compare_dental_ablation.py` on v4 val only.

Compare output:

`reports/ablation/dentex_v4_region_full_1280_20260523_152538/compare_20260523_152542_1cfc9f5`

Overall metrics:

| Experiment | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| v4 quick 50e | 0.5691 | 0.6615 | 0.6479 | 0.3859 |
| v4 full best | 0.5984 | 0.6594 | 0.6652 | 0.4065 |
| delta full - quick | +0.0293 | -0.0020 | +0.0173 | +0.0206 |

Per-class metrics:

| Class | Run | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Caries | quick | 0.5116 | 0.5000 | 0.5155 | 0.3428 |
| Caries | full | 0.4857 | 0.5678 | 0.5799 | 0.3653 |
| Caries | delta | -0.0259 | +0.0678 | +0.0643 | +0.0225 |
| Periapical_Lesion | quick | 0.6199 | 0.5812 | 0.6095 | 0.2523 |
| Periapical_Lesion | full | 0.6145 | 0.5718 | 0.5943 | 0.2573 |
| Periapical_Lesion | delta | -0.0054 | -0.0094 | -0.0152 | +0.0050 |
| Impacted | quick | 0.5759 | 0.9032 | 0.8186 | 0.5627 |
| Impacted | full | 0.6952 | 0.8387 | 0.8215 | 0.5971 |
| Impacted | delta | +0.1193 | -0.0645 | +0.0029 | +0.0343 |

## Interpretation

The full run improves the main overall validation target: `mAP50-95` increases from `0.3859` to `0.4065`, and `mAP50` also improves from `0.6479` to `0.6652`.

Periapical_Lesion remains learnable with the v4 region formulation. Its recall is essentially stable but slightly lower than quick (`0.5812 -> 0.5718`), mAP50 is slightly lower (`0.6095 -> 0.5943`), and mAP50-95 is slightly higher (`0.2523 -> 0.2573`). This suggests the full run did not collapse class 1, but it also did not clearly improve class 1 recall.

Caries improves substantially in recall and mAP, while Impacted improves precision and mAP50-95 but loses some recall. Neither Caries nor Impacted shows a severe degradation.

The training curve shows overfitting after the best epoch. EarlyStopping selected epoch `35` as best and stopped at epoch `135`. Use `best.pt`, not `last.pt`.

## Recommendation

The v4_region YOLOv8n full baseline is stronger than the quick baseline overall and is suitable for the next final validation step on the held-out test set, once approved.

Recommended next step after approval:

1. Run exactly one final test evaluation on `best.pt` using the v4 test split.
2. Do not tune hyperparameters from test results.
3. Keep this model as the current v4_region YOLOv8n baseline.

A future `v4_context` dataset may still be useful if the project prioritizes easier clinical visualization or larger context around periapical lesions, but it is not required before the first final test. The current v4_region small-lesion target is internally consistent and trains successfully.
