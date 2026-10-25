# 角点预测网络
按 `Dataset → DataLoader → nn.Module → loss / optimizer → 训练循环 → evaluation` 的顺序组织，使用普通 Python 脚本。

- [dataset.py](dataset.py)：RGB 与八通道热图标签配对，返回 `(x,y)`。
- [model.py](model.py)：ResNet-18 encoder、五次转置卷积上采样、1×1 Conv 和 Sigmoid。
- [check_model.py](check_model.py)：读取真实 batch，检查前向、反向计算，并提取八个二维角点。
- [train.py](train.py)：SmoothL1 热图损失、AdamW、训练循环和权重保存。
- [tune.py](tune.py)：固定 16/4 划分的学习率与架构对照，每轮记录 CSV。
- [evaluate.py](evaluate.py)：计算合成图的角点像素误差，输出 GT 绿色、预测红色的叠加图。
- [test_video.py](test_video.py)：完整视频逐帧测试，输出原画面和左右目标放大视频。
- [transformer_model.py](transformer_model.py)：32×32 patch、384 维、6 层、6 heads 的小型 ViT，沿用同一 decoder 和热图头。
- [EXPERIMENTS.md](EXPERIMENTS.md)：实验条件、结果与结论。

## 运行
使用已配置的 Python 环境。下面的命令在项目根目录执行：
```powershell
& 'S:\Deep Learning\eecs498\.venv\Scripts\python.exe' corner_prediction/check_model.py
```
启动小样本训练：
```powershell
& 'S:\Deep Learning\eecs498\.venv\Scripts\python.exe' corner_prediction/train.py
```
默认使用 CUDA，训练 100 epochs，batch size 为 2，learning rate 为 `1e-4`，这些参数直接写在 `train.py` 开头。训练结束保存 `corner_prediction/corner_predictor.pt`，权重由 Git 忽略。`check_model.py` 仅计算梯度，不执行参数更新。

## 数据与网络
脚本根据自身位置定位项目目录，可从其他工作目录运行。RGB 只做 ToTensor 和 ImageNet normalization，保持原始分辨率；标签保持 `[0,1]`。输入 `(N,3,480,640)`，ResNet 输出 `(N,512,15,20)`，decoder 恢复到 `(N,16,480,640)`，最后映射为 `(N,8,480,640)`。
`weights=None` 为随机初始化。当前使用 BoxDreamer 的 coarse heatmap loss，角点坐标的 fine loss 后续接入。当前高、宽都是 32 的倍数，经过五次上采样恢复原尺寸；热图生成方式见 [heatmaps_README.md](../data_prep/heatmaps_README.md)。
`check_model.py` 中，`eval()` 和 `no_grad()` 用于推理，再从每个通道取最大响应位置，得到 `(N,8,2)` 的角点坐标。该检查脚本仍使用随机初始化模型；训练后权重测试使用 `evaluate.py`、`predict.py` 或 `test_video.py`。

## 调参复现
以下命令从项目根目录运行。新实验使用新的输出目录，以保留原始记录。
```powershell
python corner_prediction/tune.py --output results/resnet_new_run
python corner_prediction/tune.py --architecture vit --learning-rates 0.0003 --output results/transformer_new_run
```
历史基线 `train.py` 仍保留 20 张全训练的设置；调参脚本使用 16 张训练、4 张验证，两者分开报告。所有视频帧仅用于测试。

## 参考
- [torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [BoxDreamer §3.3：SmoothL1 热图监督](https://arxiv.org/html/2504.07955v2#S3.SS3)
