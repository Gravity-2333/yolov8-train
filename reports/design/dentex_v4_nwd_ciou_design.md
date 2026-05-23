# DENTEX v4 NWD-CIoU Design

## Purpose

The current DENTEX v4 region baseline can detect Periapical_Lesion roughly, but its strict localization quality remains weak. The 1280 full baseline is still the main candidate, while 1536 quick and P2 quick did not clearly improve Periapical_Lesion recall or mAP50-95.

This design proposes a minimal source-level experiment:

`YOLOv8n + NWD-CIoU mixed bbox loss`

The goal is to improve small lesion box robustness without changing the dataset, model architecture, assignment logic, or default CIoU baseline behavior.

No training is started by this document.

## Background

The Normalized Gaussian Wasserstein Distance (NWD) paper models a bounding box as a 2D Gaussian distribution and uses a normalized Wasserstein similarity to reduce the excessive sensitivity of IoU to tiny box shifts. This is relevant here because v4_region class 1 uses small Periapical_Lesion region boxes. For these boxes, a few-pixel offset can produce a large IoU drop even when the prediction is clinically close.

Reference:

- Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection", arXiv:2110.13389, https://arxiv.org/abs/2110.13389

## Current Source Locations

Current YOLOv8 detection bbox loss:

- `ultralytics/utils/loss.py`
  - `BboxLoss.forward()`
  - current line of interest:
    - `iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)`
    - `loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum`

Current IoU utility:

- `ultralytics/utils/metrics.py`
  - `bbox_iou()`
  - supports IoU/GIoU/DIoU/CIoU

Current dental training entrypoint:

- `scripts/train_dental_ablation.py`
  - already has `--bbox-loss-type`
  - currently rejects anything except `ciou`

## Proposed Files To Modify

Minimal implementation should touch only:

1. `ultralytics/utils/metrics.py`
   - Add `bbox_nwd_similarity()` or similarly named helper.
   - Keep it separate from `bbox_iou()` to avoid changing existing IoU call behavior.

2. `ultralytics/utils/loss.py`
   - Add optional parameters to `BboxLoss`.
   - Use CIoU by default.
   - Use mixed NWD-CIoU only when explicitly requested.

3. `scripts/train_dental_ablation.py`
   - Allow `--bbox-loss-type nwd_ciou`.
   - Add `--nwd-ratio`, default `0.5`.
   - Add `--nwd-constant`, default initially `12.8`.
   - Pass these values to the model/loss configuration through a controlled hook.

4. Optional:
   - `tests/test_dental_nwd_loss.py`
   - Small tensor-level unit checks for shape, finite values, gradients, and CIoU default behavior.

## NWD Formula

For a box represented as `(cx, cy, w, h)`, model it as a Gaussian:

`N(cx, cy, w/2, h/2)` in simplified diagonal form.

For two boxes `A` and `B`, a commonly used closed-form squared distance is:

`W2_2(A, B) = (cx_a - cx_b)^2 + (cy_a - cy_b)^2 + ((w_a - w_b)^2 + (h_a - h_b)^2) / 4`

Then convert distance into a similarity:

`NWD = exp(-sqrt(W2_2 + eps) / C)`

where `C` is a dataset/scale normalization constant.

For YOLOv8 loss, boxes in `BboxLoss.forward()` are decoded into feature/input pixel coordinates, not normalized 0..1 labels. Therefore `C` must be chosen for pixel-space boxes. Start conservatively with:

`C = 12.8`

This is a tunable constant. If the gradients look too weak or too sharp in sanity checks, test `C` in `{8.0, 12.8, 16.0, 24.0}` before long training. Do not tune on test.

## Mixed Loss

Keep the current CIoU path intact:

`ciou = bbox_iou(pred, target, xywh=False, CIoU=True)`

Compute NWD similarity:

`nwd = bbox_nwd_similarity(pred, target, xywh=False, constant=C)`

Build mixed similarity:

`mixed = (1 - ratio) * ciou + ratio * nwd`

Then use the same weighting pattern:

`loss_iou = ((1.0 - mixed) * weight).sum() / target_scores_sum`

Recommended first experiment:

`ratio = 0.5`

This is intentionally a mixed loss, not pure NWD. CIoU remains useful for shape, center, and larger object alignment. NWD should soften the harsh penalty for tiny lesion localization shifts.

## Default Behavior Guarantee

Default training must remain unchanged:

- `--bbox-loss-type ciou`
- `--nwd-ratio` ignored
- `--nwd-constant` ignored
- `BboxLoss` defaults to CIoU behavior
- Official model YAMLs still build
- Existing v4 1280 full/quick comparisons remain fair

The implementation should make `nwd_ciou` opt-in only.

## Implementation Hook Design

Ultralytics creates `BboxLoss` inside `v8DetectionLoss.__init__()`:

`self.bbox_loss = BboxLoss(m.reg_max).to(device)`

Minimal options:

### Option A: Global runtime config on model args

1. In `scripts/train_dental_ablation.py`, pass extra training kwargs:
   - `bbox_loss_type=args.bbox_loss_type`
   - `nwd_ratio=args.nwd_ratio`
   - `nwd_constant=args.nwd_constant`

2. If Ultralytics args validation rejects unknown keys, do not use this path.

### Option B: Environment variables

Set process-scoped env vars in `train_dental_ablation.py` before `model.train()`:

- `YOLO_DENTAL_BBOX_LOSS_TYPE=nwd_ciou`
- `YOLO_DENTAL_NWD_RATIO=0.5`
- `YOLO_DENTAL_NWD_CONSTANT=12.8`

Then read them in `BboxLoss.__init__()`.

This is less elegant but low risk for this project because it is contained to the launched training process and does not modify Ultralytics CLI schema.

### Preferred first implementation

Use Option B first for minimal intrusion:

- It avoids broad changes to trainer/config validation.
- It preserves existing model YAMLs.
- It keeps the experiment clearly controlled by `scripts/train_dental_ablation.py`.

If later this becomes a reusable feature, refactor to a proper config/args path.

## Pseudocode

In `ultralytics/utils/metrics.py`:

```python
def bbox_nwd_similarity(box1, box2, xywh=False, constant=12.8, eps=1e-7):
    if not xywh:
        x1, y1, x2, y2 = box1.chunk(4, -1)
        x1g, y1g, x2g, y2g = box2.chunk(4, -1)
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        cx2, cy2 = (x1g + x2g) / 2, (y1g + y2g) / 2
        w1, h1 = (x2 - x1).clamp_min(eps), (y2 - y1).clamp_min(eps)
        w2, h2 = (x2g - x1g).clamp_min(eps), (y2g - y1g).clamp_min(eps)
    else:
        cx1, cy1, w1, h1 = box1.chunk(4, -1)
        cx2, cy2, w2, h2 = box2.chunk(4, -1)
        w1, h1 = w1.clamp_min(eps), h1.clamp_min(eps)
        w2, h2 = w2.clamp_min(eps), h2.clamp_min(eps)

    wasserstein_2 = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2) + ((w1 - w2).pow(2) + (h1 - h2).pow(2)) / 4
    return torch.exp(-torch.sqrt(wasserstein_2 + eps) / constant)
```

In `ultralytics/utils/loss.py`:

```python
ciou = bbox_iou(..., CIoU=True)
if self.bbox_loss_type == "nwd_ciou":
    nwd = bbox_nwd_similarity(..., constant=self.nwd_constant)
    similarity = (1.0 - self.nwd_ratio) * ciou + self.nwd_ratio * nwd
else:
    similarity = ciou
loss_iou = ((1.0 - similarity) * weight).sum() / target_scores_sum
```

Clamp ratio to `[0, 1]` at initialization.

## Checks Before Training

Run these before any GPU training:

```bash
python -m py_compile ultralytics/utils/metrics.py ultralytics/utils/loss.py scripts/train_dental_ablation.py
```

Tensor-level unit check:

```bash
python - <<'PY'
import torch
from ultralytics.utils.metrics import bbox_nwd_similarity

p = torch.tensor([[10., 10., 20., 20.]], requires_grad=True)
t = torch.tensor([[11., 11., 21., 21.]])
s = bbox_nwd_similarity(p, t, xywh=False, constant=12.8)
assert s.shape == (1, 1)
assert torch.isfinite(s).all()
(1 - s).sum().backward()
assert torch.isfinite(p.grad).all()
print("nwd check ok", s.item())
PY
```

Default behavior check:

```bash
python - <<'PY'
from ultralytics import YOLO
for p in [
    "ultralytics/cfg/models/v8/yolov8n.yaml",
    "ultralytics/cfg/models/v8/yolov8s.yaml",
    "ultralytics/cfg/models/v8/yolov8m.yaml",
]:
    m = YOLO(p)
    m.info()
print("official model build ok")
PY
```

## 1 Epoch Sanity Plan

Only after implementation and local checks:

```bash
screen -S dentex_v4_nwd_sanity
cd /root/autodl-tmp/work/yolov8-train

python scripts/train_dental_ablation.py \
  --data ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml \
  --model yolov8n.pt \
  --weights yolov8n.pt \
  --imgsz 640 \
  --epochs 1 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --name dentex_v4_region_yolov8n_nwd_ciou_640_sanity \
  --project /root/autodl-tmp/work/yolov8-train/runs/detect/dentex_v4_region_nwd_sanity \
  --bbox-loss-type nwd_ciou \
  --nwd-ratio 0.5 \
  --nwd-constant 12.8
```

Pass criteria:

- no OOM
- no NaN
- no path/class error
- no loss shape error
- `results.csv`, `best.pt`, `last.pt`, and `run_metadata.yaml` exist
- CIoU default still works in a separate 1 epoch smoke if needed

## 50 Epoch Quick Command

Only after sanity passes and user confirms:

```bash
screen -S dentex_v4_nwd_quick
cd /root/autodl-tmp/work/yolov8-train

python scripts/train_dental_ablation.py \
  --data ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml \
  --model yolov8n.pt \
  --weights yolov8n.pt \
  --imgsz 1280 \
  --epochs 50 \
  --batch 16 \
  --device 0 \
  --workers 24 \
  --name dentex_v4_region_yolov8n_nwd_ciou_1280_quick_b16 \
  --project /root/autodl-tmp/work/yolov8-train/runs/detect/dentex_v4_region_nwd_quick \
  --bbox-loss-type nwd_ciou \
  --nwd-ratio 0.5 \
  --nwd-constant 12.8
```

Compare only on val:

```bash
python scripts/compare_dental_ablation.py \
  --data ultralytics/cfg/datasets/dentex_yolov8_3cls_v4_region_server.yaml \
  --weights \
    /path/to/v4_1280_quick_baseline_best.pt \
    /path/to/nwd_ciou_quick_best.pt \
  --names \
    dentex_v4_region_yolov8n_1280_quick_b16 \
    dentex_v4_region_yolov8n_nwd_ciou_1280_quick_b16 \
  --imgsz 1280 \
  --conf 0.001 \
  --iou 0.7 \
  --device 0 \
  --out-dir reports/ablation/dentex_v4_region_nwd_ciou_quick_$(date +%Y%m%d_%H%M%S)
```

Do not run test.

## Risks

1. `C` sensitivity
   - If `C` is too large, NWD becomes too forgiving and localization pressure weakens.
   - If `C` is too small, it becomes sharp and may not solve tiny-box sensitivity.

2. Larger objects
   - Caries and Impacted may rely more on ordinary IoU geometry.
   - A high NWD ratio could reduce their localization quality.
   - This is why the first ratio should be `0.5`, not pure NWD.

3. Assignment remains IoU-based
   - This proposal only changes bbox regression loss.
   - It does not change TaskAlignedAssigner positive-sample assignment.
   - If recall remains poor, future work may consider NWD-aware assignment, but not in the first experiment.

4. Evaluation metric remains IoU mAP
   - Training with NWD may improve robustness without always improving IoU mAP50-95.
   - The decision should still be based on standard val mAP and per-class recall, because final reporting uses YOLO metrics.

5. Default behavior contamination
   - Any implementation mistake that changes CIoU default would invalidate baseline comparisons.
   - Therefore CIoU default must be tested before NWD training.

## Why NWD-CIoU Is More Targeted Than More P2 Or Attention

P2 quick increased small-scale detection capacity but did not improve Periapical_Lesion recall. The 1536 quick increased input resolution but also failed to improve Periapical_Lesion recall and strict localization. These results suggest the bottleneck is not only feature-map scale.

The v4_region Periapical_Lesion boxes are small lesion-region boxes. The model often needs robust localization under tiny pixel shifts. NWD directly targets this failure mode by making bbox similarity less brittle for tiny boxes, while retaining CIoU in the mixed loss to preserve ordinary box geometry.

Therefore NWD-CIoU is a better next design target than stacking more architecture changes at this stage.

## Recommendation

Do not implement or train immediately without confirmation.

Recommended next step:

1. Implement `bbox_nwd_similarity()`.
2. Add opt-in `nwd_ciou` path to `BboxLoss`.
3. Extend `scripts/train_dental_ablation.py` with `--nwd-ratio` and `--nwd-constant`.
4. Run py_compile and tensor checks.
5. Run 1 epoch sanity.
6. If sanity passes, run one 50 epoch NWD-CIoU quick.

Keep the current main candidate unchanged until val evidence shows otherwise:

`v4_region YOLOv8n 1280 full best.pt`
