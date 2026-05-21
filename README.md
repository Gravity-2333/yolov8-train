# YOLOv8 Train

This repository is my learning and training workspace for YOLO-based dental lesion region detection.

It is used to study YOLO training workflows, prepare dental X-ray lesion datasets, run baseline experiments, and later improve the model for lesion region recognition.

The current focus is a clean three-class lesion detection baseline:

- `0`: Caries
- `1`: Periapical Lesion
- `2`: Impacted

## Purpose

This repo keeps the YOLOv8 source code and training configuration used for dental X-ray lesion experiments. It is not an upstream Ultralytics mirror; it is a project workspace for learning, training, and model improvement.

Dataset files, large model weights, training runs, and exported archives should be stored outside the repository unless they are intentionally small metadata files.

## Recommended Training Command

Example baseline command:

```bash
yolo detect train \
  model=yolov8n.pt \
  data=/root/autodl-tmp/work/yolo8_training/data_v1.yaml \
  imgsz=1024 \
  epochs=200 \
  batch=64 \
  workers=8 \
  device=0 \
  cache=ram \
  amp=True \
  cos_lr=True \
  plots=False
```

For larger YOLOv8 variants, reduce `batch` according to GPU memory.

## Project Notes

- Keep source changes separate from dataset conversion and training output.
- Do not commit raw datasets, `.tar` archives, or full training result folders.
- Use `screen` or a similar terminal multiplexer for long remote training jobs.
- Save checkpoints outside the source tree when possible.

## Upstream

This repository is derived from the Ultralytics YOLO source code. The original project is available at:

https://github.com/ultralytics/ultralytics

The source license is AGPL-3.0. See `LICENSE` for details.
