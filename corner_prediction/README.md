# 角点预测网络
按 `Dataset → DataLoader → nn.Module → loss / optimizer → 训练循环 → evaluation` 的顺序组织，使用普通 Python 脚本。

- [dataset.py](dataset.py)：RGB 与八通道热图标签配对，返回 `(x,y)`。
- [model.py](model.py)：ResNet-18 encoder、五次转置卷积上采样、1×1 Conv 和 Sigmoid。
- [check_model.py](check_model.py)：读取真实 batch，检查前向、反向计算，并提取八个二维角点。
- [train.py](train.py)：SmoothL1 热图损失、AdamW、训练循环和权重保存。

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
`check_model.py` 中，`eval()` 和 `no_grad()` 用于推理，再从每个通道取最大响应位置，得到 `(N,8,2)` 的角点坐标。当前仅完成结构与计算检查，网络尚未正式训练。

## 参考
- [torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [BoxDreamer §3.3：SmoothL1 热图监督](https://arxiv.org/html/2504.07955v2#S3.SS3)
