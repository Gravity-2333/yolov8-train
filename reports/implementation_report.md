# Dental NeckPlus v1 Implementation Report

## Current State

- Branch: `feat/dental-neckplus-v1`
- Base commit at branch creation: `bd0f1a161`
- Purpose: first-stage, minimal runnable YOLOv8 dental lesion ablation framework.
- This stage does not run local CPU training and does not submit weights, datasets, caches, archives, or `runs/`.

## Files Changed

- Added dataset yaml templates:
  - `ultralytics/cfg/datasets/dental_lesion_3cls_large_server.yaml`
  - `ultralytics/cfg/datasets/dental_lesion_3cls_large_example.yaml`
- Added scripts:
  - `scripts/dataset_audit_dental.py`
  - `scripts/train_dental_ablation.py`
  - `scripts/compare_dental_ablation.py`
- Added model modules:
  - `ultralytics/nn/modules/dental_neck.py`
- Added model YAMLs:
  - `ultralytics/cfg/models/v8/yolov8n_dental_eca_p3.yaml`
  - `ultralytics/cfg/models/v8/yolov8n_dental_bifpn_eca.yaml`
- Updated registration:
  - `ultralytics/nn/modules/__init__.py`
  - `ultralytics/nn/tasks.py`
- Updated `.gitignore` to guard against training outputs, datasets, weights, archives, and cache files.

## Modules

`DentalECA(c1, k_size=3)` keeps `[B, C, H, W]` unchanged. It uses global average pooling, a tiny `Conv1d` channel interaction, sigmoid weights, and channel reweighting.

`BiFPN_Add2(c1_list, c2, eps=1e-4, resize_mode="nearest")` and `BiFPN_Add3(...)` fuse two or three feature maps. The first `from` input defines target H,W. Other inputs are resized to that size, channels are projected to `c2` with 1x1 Conv when needed, positive learned weights are normalized with `relu(w) / (sum(w) + eps)`, and the fused output passes through Conv.

`WeightedConcat` is intentionally not implemented in this stage. It remains a later fallback if Add-style fusion proves unstable.

## parse_model Changes

The parser adds narrow cases for the new modules only:

- `DentalECA`: single input, output channels equal input channels.
- `BiFPN_Add2/Add3`: `from` must be a list of 2 or 3 inputs. YAML `args[0]` is the output channel count and is width-scaled like other YOLOv8 channels.

Official YOLOv8 modules and existing YAML behavior are otherwise left unchanged.

## Model YAMLs

`yolov8n_dental_eca_p3.yaml` keeps the official YOLOv8n backbone and neck, then applies `DentalECA` only to P3 before `Detect`. It is the smallest attention ablation.

`yolov8n_dental_bifpn_eca.yaml` keeps the official backbone and replaces key neck fusions with lightweight weighted Add fusion. It outputs P3/P4/P5 to `Detect` and applies `DentalECA` to those detection features.

Both YAMLs keep `nc: 80`; Ultralytics overrides it from the dataset yaml during training.

## Dataset YAML

Server dataset path:

```yaml
path: /root/autodl-tmp/work/datasets_yolo8/dental_lesion_3cls_large_v1
train: images/train
val: images/val
test: images/test
names:
  0: Caries
  1: Periapical Lesion
  2: Deep Caries
```

The requested prompt names class 2 as `Deep Caries`. If the actual cleaned v1 dataset uses `Impacted` for class 2, update the dataset YAML before training or document the mismatch in the experiment notes. Do not change the dataset files in place.

## Local Test Commands

```bash
mamba run -n yolo python -m py_compile scripts/dataset_audit_dental.py scripts/train_dental_ablation.py scripts/compare_dental_ablation.py ultralytics/nn/modules/dental_neck.py
mamba run -n yolo python -c "from ultralytics.nn.modules.dental_neck import DentalECA, BiFPN_Add2, BiFPN_Add3; print('ok')"
mamba run -n yolo python -c "from ultralytics import YOLO; YOLO('ultralytics/cfg/models/v8/yolov8n_dental_eca_p3.yaml').model.info(verbose=True, imgsz=640)"
mamba run -n yolo python -c "from ultralytics import YOLO; YOLO('ultralytics/cfg/models/v8/yolov8n_dental_bifpn_eca.yaml').model.info(verbose=True, imgsz=640)"
```

Also verify official YAMLs still build:

```bash
mamba run -n yolo python -c "from ultralytics import YOLO; YOLO('ultralytics/cfg/models/v8/yolov8.yaml').model.info(verbose=True, imgsz=640)"
mamba run -n yolo python -c "from ultralytics import YOLO; YOLO('ultralytics/cfg/models/v8/yolov8n.yaml').model.info(verbose=True, imgsz=640)"
mamba run -n yolo python -c "from ultralytics import YOLO; YOLO('ultralytics/cfg/models/v8/yolov8s.yaml').model.info(verbose=True, imgsz=640)"
mamba run -n yolo python -c "from ultralytics import YOLO; YOLO('ultralytics/cfg/models/v8/yolov8m.yaml').model.info(verbose=True, imgsz=640)"
```

Executed local validation uses the `mamba` environment named `yolo`; do not install test-only packages into the global Python environment.

## Server Sanity Train

Clone or update:

```bash
git clone -b feat/dental-neckplus-v1 https://github.com/Gravity-2333/yolov8-train.git
cd yolov8-train
pip install -e .
```

For an existing clone:

```bash
git fetch
git checkout feat/dental-neckplus-v1
git pull
pip install -e .
```

1 epoch sanity commands:

```bash
python scripts/train_dental_ablation.py \
  --data ultralytics/cfg/datasets/dental_lesion_3cls_large_server.yaml \
  --model yolov8n.pt \
  --weights yolov8n.pt \
  --imgsz 640 --epochs 1 --batch 8 --device 0 --workers 8 \
  --name baseline_yolov8n_640_sanity \
  --project runs/detect/dental_neckplus \
  --bbox-loss-type ciou

python scripts/train_dental_ablation.py \
  --data ultralytics/cfg/datasets/dental_lesion_3cls_large_server.yaml \
  --model ultralytics/cfg/models/v8/yolov8n_dental_eca_p3.yaml \
  --weights yolov8n.pt \
  --imgsz 640 --epochs 1 --batch 8 --device 0 --workers 8 \
  --name yolov8n_dental_eca_p3_640_sanity \
  --project runs/detect/dental_neckplus \
  --bbox-loss-type ciou

python scripts/train_dental_ablation.py \
  --data ultralytics/cfg/datasets/dental_lesion_3cls_large_server.yaml \
  --model ultralytics/cfg/models/v8/yolov8n_dental_bifpn_eca.yaml \
  --weights yolov8n.pt \
  --imgsz 640 --epochs 1 --batch 8 --device 0 --workers 8 \
  --name yolov8n_dental_bifpn_eca_640_sanity \
  --project runs/detect/dental_neckplus \
  --bbox-loss-type ciou
```

Sanity pass means model builds, pretrained weights load or partially load, one train epoch finishes, validation runs, `results.csv` exists, best/last weights exist, and losses are not NaN.

## Quick Ablation

Run only after sanity passes:

```bash
python scripts/train_dental_ablation.py \
  --data ultralytics/cfg/datasets/dental_lesion_3cls_large_server.yaml \
  --model ultralytics/cfg/models/v8/yolov8n_dental_eca_p3.yaml \
  --weights yolov8n.pt \
  --imgsz 960 --epochs 50 --batch 16 --device 0 --workers 16 \
  --name yolov8n_dental_eca_p3_960_quick \
  --project runs/detect/dental_neckplus \
  --bbox-loss-type ciou
```

Compare several finished weights:

```bash
python scripts/compare_dental_ablation.py \
  --data ultralytics/cfg/datasets/dental_lesion_3cls_large_server.yaml \
  --weights path/to/best1.pt path/to/best2.pt \
  --names baseline eca_p3 \
  --imgsz 960 --device 0
```

## Rollback

To return to official YOLOv8 behavior, train with official YAML or weights only:

```bash
python scripts/train_dental_ablation.py \
  --data ultralytics/cfg/datasets/dental_lesion_3cls_large_server.yaml \
  --model yolov8n.pt --weights yolov8n.pt
```

The official `yolov8.yaml`, `yolov8n.yaml`, `yolov8s.yaml`, and `yolov8m.yaml` are not modified.

## Not Included In Stage 1

- WIoU/PIoU is not connected to the main training loss.
- `--bbox-loss-type` only accepts `ciou`.
- m-version model YAMLs are not implemented.
- SAHI is not required for training, validation, or comparison.
- `WeightedConcat` is not implemented.

## Known Risks

- Custom YAMLs may only partially load `yolov8n.pt`; check transfer counts in server logs.
- If the actual dataset class 2 is `Impacted`, the current prompt-provided `Deep Caries` name should be corrected before reporting experiments.
- `imgsz=960` can require lowering batch before reducing image size.

## Files That Must Not Be Committed

Do not commit `runs/`, `datasets/`, `transfer/`, `wandb/`, `*.pt`, `*.pth`, `*.onnx`, `*.engine`, `*.tar`, `*.tar.gz`, `*.zip`, `*.cache`, `__pycache__/`, or generated training artifacts.
