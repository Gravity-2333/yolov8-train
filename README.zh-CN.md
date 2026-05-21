# YOLOv8 Train

这是我用来学习和训练基于 YOLO 的牙齿病变区域识别项目的仓库。

本仓库用于学习 YOLO 训练流程、整理牙科 X 光病变检测数据集、运行 baseline 实验，并在后续继续改进牙齿病变区域识别模型。

当前主要任务是三分类牙科 X 光病变检测 baseline：

- `0`: Caries
- `1`: Periapical Lesion
- `2`: Impacted

## 用途

本仓库只保留 YOLOv8 源码、训练配置和必要脚本。它不是 Ultralytics 上游镜像，而是用于学习、训练和改进模型的项目工作区。

原始数据集、大模型权重、训练输出目录、导出压缩包等大文件不建议提交到 GitHub。

## 推荐训练命令

baseline 示例：

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

如果换成 YOLOv8s 或 YOLOv8m，需要根据显存适当降低 `batch`。

## 项目约定

- 源码改动、数据转换、训练输出尽量分开管理。
- 不要提交原始数据集、`.tar` 数据包或完整训练结果目录。
- 远程长时间训练建议使用 `screen` 后台运行。
- checkpoint 和导出模型尽量放在源码仓库外部。

## 上游来源

本仓库基于 Ultralytics YOLO 源码整理，原始项目地址：

https://github.com/ultralytics/ultralytics

源码许可证为 AGPL-3.0，详见 `LICENSE`。
