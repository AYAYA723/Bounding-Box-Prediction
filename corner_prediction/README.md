# 角点预测网络
入口为 [corner_prediction.ipynb](corner_prediction.ipynb)，按现有 `pytorch 基础.md` 的顺序组织：Dataset、DataLoader、nn.Module、loss/optimizer、训练循环、evaluation。`model.py` 定义网络，`dataset.py` 配对 RGB 和八通道热图。

模型采用 **ResNet-18 encoder → 五次转置卷积上采样 → 1×1 Conv → Sigmoid**。输入 `(N,3,480,640)`，输出 `(N,8,480,640)`，与现有热图标签保持相同分辨率和角点顺序。当前用 SmoothL1 热图损失和 AdamW；角点坐标的 fine loss 后续接入。

## 运行
在 notebook 右上角选择现有 Windows Python 环境：
```text
S:\Deep Learning\eecs498\.venv\Scripts\python.exe
```
该环境已有 PyTorch、torchvision、NumPy、Pillow 和 RTX 5070 的 CUDA 支持。Notebook 使用明确的 S 盘项目路径；`weights=None` 不下载权重。所有可执行代码单元只检查数据读取、前向、反向和角点输出格式，训练循环与保存模型示例放在 Markdown 代码块中。

RGB 只做 ToTensor 和 ImageNet normalization，不做 resize/crop，标签保持 `[0,1]`。当前高宽都是 32 的倍数，经 encoder 下采样 32 倍，再由 decoder 恢复原尺寸。标签本身的生成说明见 [heatmaps_README.md](../data_prep/heatmaps_README.md)。

## 检查结果
已在现有 RTX 5070 上执行 notebook 的全部 6 个代码单元，读取真实图片与标签；batch 为 2 时输出 `(2,8,480,640)`，单张推理输出 `(1,8,480,640)`，角点坐标形状为 `(2,8,2)`。全部可训练参数的梯度有限且 encoder、head 的梯度非零。参数量为 13,971,608，本次检查峰值 CUDA 分配约 593 MiB。检查没有执行 optimizer step，当前尚无训练后的预测结果。本机检查记录为 `verification.json`，作为生成结果保留在本地。

## 参考
- [torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [BoxDreamer §3.3：SmoothL1 热图监督](https://arxiv.org/html/2504.07955v2#S3.SS3)
