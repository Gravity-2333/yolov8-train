# DENTEX v4 Region YOLOv8n-P2 Quick Ablation Report

## Scope

This ablation tests a lightweight YOLOv8n-P2 detector for v4_region small lesion detection. The test set was not used.

Constraints followed:

- no test evaluation
- no YOLOv8m
- no ECA/BiFPN
- no WIoU/PIoU
- no SAHI
- no balanced sampling
- no model weights or large result packages pulled to local

## Model YAML

New model config:

`ultralytics/cfg/models/v8/yolov8n_dental_p2.yaml`

The config is based on the official `yolov8-p2.yaml` structure and only changes the task classes to the v4_region dental classes.

Classes:

- `0 Caries`
- `1 Periapical_Lesion`
- `2 Impacted`

Detect outputs:

- P2/4
- P3/8
- P4/16
- P5/32

No Python modules were changed.

## Model Structure

Build and dummy forward passed.

| Model | Params | GFLOPs | Detect heads |
| --- | ---: | ---: | --- |
| YOLOv8n baseline | 3,006,233 | 32.35 | P3/P4/P5 |
| YOLOv8n-P2 | 2,921,436 fused / 2,926,956 unfused | 48.75 compare / 12.4 build summary | P2/P3/P4/P5 |

The GFLOPs reported by `compare_dental_ablation.py` is used for the fair comparison table.

## Sanity

P2 1-epoch sanity passed.

- sanity run: `dentex_v4_region_yolov8n_p2_640_sanity_20260523_155958_7dc16b6`
- imgsz: `640`
- batch: `8`
- epochs: `1`
- status: completed
- artifacts: `results.csv`, `args.yaml`, `run_metadata.yaml`, `best.pt`, `last.pt`
- errors: no OOM, no NaN, no channel mismatch, no Detect shape error, no path error, no class error

## Quick Training

P2 quick completed successfully.

- run: `dentex_v4_region_yolov8n_p2_1280_quick_b16_20260523_160309_7dc16b6`
- imgsz: `1280`
- epochs: `50`
- batch: `16`
- workers: `24`
- bbox loss type: `ciou`
- cache: disabled
- plots: disabled by training script
- batch fallback: not needed, batch remained `16`
- GPU peak observed from runtime sampling: about `13.5 GB`
- errors: no OOM, no NaN, no channel mismatch, no path error, no class error

Best epoch from `results.csv`: `29` by mAP50-95.

## Unified Val Compare

Compare output:

`reports/ablation/dentex_v4_region_p2_quick_compare_20260523_163500/compare_20260523_163504_7dc16b6`

Both models were validated on the same v4 val split with:

- imgsz: `1280`
- conf: `0.001`
- iou: `0.7`
- device: `0`

Overall metrics:

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n quick baseline | 0.5691 | 0.6615 | 0.6479 | 0.3859 |
| YOLOv8n-P2 quick | 0.6014 | 0.6082 | 0.6231 | 0.3867 |
| delta P2 - baseline | +0.0323 | -0.0533 | -0.0247 | +0.0008 |

Per-class metrics:

| Class | Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Caries | baseline | 0.5116 | 0.5000 | 0.5155 | 0.3428 |
| Caries | P2 | 0.4190 | 0.5932 | 0.5018 | 0.3511 |
| Caries | delta | -0.0926 | +0.0932 | -0.0137 | +0.0083 |
| Periapical_Lesion | baseline | 0.6199 | 0.5812 | 0.6095 | 0.2523 |
| Periapical_Lesion | P2 | 0.6804 | 0.4894 | 0.6120 | 0.2728 |
| Periapical_Lesion | delta | +0.0605 | -0.0918 | +0.0025 | +0.0205 |
| Impacted | baseline | 0.5759 | 0.9032 | 0.8186 | 0.5627 |
| Impacted | P2 | 0.7049 | 0.7419 | 0.7556 | 0.5362 |
| Impacted | delta | +0.1289 | -0.1613 | -0.0630 | -0.0265 |

## Interpretation

P2 did not produce a clear win for the current objective.

Positive signals:

- Overall mAP50-95 is marginally higher: `0.3859 -> 0.3867`.
- Periapical_Lesion precision improves: `0.6199 -> 0.6804`.
- Periapical_Lesion mAP50-95 improves: `0.2523 -> 0.2728`.
- Caries recall improves: `0.5000 -> 0.5932`.

Negative signals:

- Overall recall drops: `0.6615 -> 0.6082`.
- Overall mAP50 drops: `0.6479 -> 0.6231`.
- Periapical_Lesion recall drops clearly: `0.5812 -> 0.4894`.
- Impacted recall and mAP both drop.
- P2 inference is slower and GFLOPs are higher in the unified compare.

For this project, the stated goal of P2 was to improve small Periapical_Lesion detection. It improves localization quality at stricter IoU ranges, as seen in class 1 mAP50-95, but it hurts class 1 recall. That is not a strong enough result to justify moving P2 directly into full training as the next mainline model.

## Recommendation

Do not start P2 full training yet.

Keep `v4_region YOLOv8n 1280 full best.pt` as the current final candidate. P2 can remain as an experimental branch because it gives a small Periapical_Lesion mAP50-95 gain, but it should not replace the baseline unless the next project goal explicitly prioritizes precision/localization quality over recall.

Recommended next step remains:

1. Run one final test evaluation on the current v4_region YOLOv8n full baseline `best.pt`, if final confirmation is approved.
2. Do not tune from test results.

Potential future optimization directions before revisiting P2 full:

- threshold analysis on val for Periapical_Lesion recall/precision tradeoff
- class-specific confidence reporting for inference
- data-level improvements around class 1 small lesions
- context-box variant only if the target definition changes from lesion-region to larger contextual region
