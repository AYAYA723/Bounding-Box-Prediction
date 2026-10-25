# 角点预测网络
按 `Dataset → DataLoader → nn.Module → loss / optimizer → 训练循环 → evaluation` 组织，核心代码直接放在本目录。

```text
predict_corner/
├── model.py                 ResNet-18、decoder 和热图输出头
├── transformer_model.py     小型 ViT，沿用同一 decoder
├── dataset.py               RGB、热图与二维角点配对
├── supervision.py           soft-argmax、坐标归一化与 loss
├── train.py                 训练循环与逐轮记录
├── tools/                   检查、调参、评估、图片和视频预测
├── data_prep/               模型准备、渲染、投影、热图和打包
├── tests/                   loss、梯度和热图公式单测
├── docs/                    实验记录与论文实现对照
└── checkpoints/             本地权重，不纳入 Git
```

## 运行
以下命令均在项目根目录 `coding_test_for_research/` 执行，`python` 使用已有的 PyTorch 环境：
```powershell
python -m predict_corner.tools.check_model
python -m predict_corner.train
```
`check_model` 读取真实 batch，检查前向、反向、fine 独立梯度和输出形状。`train.py` 默认 CUDA、100 epochs、batch size=2、learning rate=`1e-4`、AdamW，参数直接写在文件开头。
每次训练的配置与三项 loss 保存在 `results/paper_train/时间/`；结束后保存该目录的 `last.pt` 和 `predict_corner/checkpoints/corner_predictor_paper.pt`。

## 数据与网络
[dataset.py](dataset.py) 返回 `(image,heatmaps,corners_xy)`。RGB 做 ToTensor 和 ImageNet normalization，保持原始分辨率；热图标签保持 `[0,1]`。热图生成方式见 [heatmaps_README.md](data_prep/heatmaps_README.md)。
[model.py](model.py) 将 `(N,3,480,640)` 编码为 `(N,512,15,20)`，五次上采样恢复到 `(N,16,480,640)`，再输出 `(N,8,480,640)`。`weights=None` 表示随机初始化。
[transformer_model.py](transformer_model.py) 使用 32×32 patch、384 维、6 层、6 heads 的小型 ViT，随后投影到 512 个通道并接相同 decoder。
两个模型均返回 `(heatmaps,corners_xy)`。坐标来自 pre-Sigmoid logits 的空间 soft-argmax，训练与推理共用此算法。[supervision.py](supervision.py) 计算 `coarse + 2*fine`，fine 比较按 `[W,H]` 归一化后的预测与 GT 坐标。实现细节见 [BOXDREAMER_ALIGNMENT.md](docs/BOXDREAMER_ALIGNMENT.md)。

## 调参、预测与测试
调参使用固定 16 张训练、4 张验证；基础 `train.py` 使用全部 20 张。新运行使用独立输出目录，视频帧仅用于测试：
```powershell
python -m predict_corner.tools.tune --output results/resnet_new_run
python -m predict_corner.tools.tune --architecture vit --learning-rates 0.0003 --output results/transformer_new_run
python -m predict_corner.tools.evaluate --weights predict_corner/checkpoints/corner_predictor_paper.pt
python -m predict_corner.tools.predict --device cuda
python -m predict_corner.tools.test_video --device cuda
python -m unittest discover -s predict_corner/tests -v
```
评估、图片预测和视频测试默认读取新训练的 `corner_predictor_paper.pt`；该权重需要先运行训练生成。可用 `--weights` 指定其他权重，ViT 同时指定 `--architecture vit`。各工具参数可用 `--help` 查看。
历史实验与视频记录见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。三个历史权重放在 `checkpoints/`，其训练和角点解码方式对应旧版，不能将其直接视为完整监督的结果。

## 参考
- [torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [BoxDreamer §3.3](https://arxiv.org/html/2504.07955v2#S3.SS3)
