# 角点预测网络
按 `Dataset → DataLoader → nn.Module → loss / optimizer → 训练循环 → evaluation` 组织，核心代码直接放在本目录。

```text
predict_corner/
├── model.py                 ResNet-18、decoder 和热图输出头
├── transformer_model.py     小型 ViT，沿用同一 decoder
├── dino_model.py            冻结的 DINOv2 特征提取器与角点网络
├── heatmap_decoder.py       Transformer、Linear 与 unpatchify
├── feature_dataset.py       冻结特征缓存与二维角点配对
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
本轮 DINOv2 + Transformer 已完成首轮 20 epochs，验证平均角点误差为 38.98 px。完整记录、架构图和续训说明见 [DINO_TRANSFORMER.md](docs/DINO_TRANSFORMER.md)。对应训练目录为 `results/dino_transformer_training/`，保存 `best.pt`、`last.pt`、`training_state.pt`。追加训练示例：
```powershell
python -m predict_corner.tools.train_dino --resume results/dino_transformer_training --epochs 20 --learning-rate 0.00003 --output results/new_dino_continuation
```
旧 ResNet 和小 ViT 基线仍可使用以下入口。
以下命令均在项目根目录 `coding_test_for_research/` 执行，`python` 使用已有的 PyTorch 环境：
```powershell
python -m predict_corner.tools.check_model
python -m predict_corner.train
```
`check_model` 读取真实 batch，检查前向、反向、fine 独立梯度和输出形状。`train.py` 默认 CUDA、100 epochs、batch size=2、learning rate=`1e-4`、AdamW，参数直接写在文件开头。
每次训练的配置与三项 loss 保存在 `results/paper_train/时间/`；结束后保存该目录的 `last.pt` 和 `predict_corner/checkpoints/corner_predictor_paper.pt`。

## 数据与网络
新增的单图 DINOv2 + Transformer 方案见 [DINO_TRANSFORMER.md](docs/DINO_TRANSFORMER.md)：冻结 DINOv2 ViT-B/14 通用预训练特征，随机初始化并训练 12 层 Transformer + Linear/unpatchify 热图 decoder，入口为 `python -m predict_corner.tools.train_dino --output results/new_dino_training`。推理时指定 `--architecture dino` 和本次训练的权重。
[dataset.py](dataset.py) 返回 `(image,heatmaps,corners_xy)`。RGB 做 ToTensor 和 ImageNet normalization，保持原始分辨率；热图标签保持 `[0,1]`。热图生成方式见 [heatmaps_README.md](data_prep/heatmaps_README.md)。
[model.py](model.py) 将 `(N,3,480,640)` 编码为 `(N,512,15,20)`，五次上采样恢复到 `(N,16,480,640)`，再输出 `(N,8,480,640)`。`weights=None` 表示随机初始化。
[transformer_model.py](transformer_model.py) 使用 32×32 patch、384 维、6 层、6 heads 的小型 ViT，随后投影到 512 个通道并接相同 decoder。
两个模型均返回 `(heatmaps,corners_xy)`。坐标来自 pre-Sigmoid logits 的空间 soft-argmax，当前默认温度为 `0.25`，训练与推理共用此设置。GT 热图回代的全部 160 个角点误差均小于 0.5 px。[supervision.py](supervision.py) 仍计算 `coarse + 2*fine`，fine 比较按 `[W,H]` 归一化后的预测与 GT 坐标。实现细节见 [BOXDREAMER_ALIGNMENT.md](docs/BOXDREAMER_ALIGNMENT.md)。

## 小改动修复与拟合检查
保留原来的 ResNet/ViT、转置卷积 decoder 和 BatchNorm。预训练只改变初始化；`AppearanceDataset` 只改变背景、颜色和轻微模糊，不移动角点。实验脚本放在 `tools/`：
```powershell
python -m predict_corner.tools.fit --images 1 --steps 2000 --output results/new_single_image
python -m predict_corner.tools.fit --images 16 --steps 5000 --eval-every 200 --pretrained --augment --target-error 2 --max-error 5 --output results/new_pretrained_appearance
```
`fit.py` 记录每一步的温度、学习率、三项 loss、定期训练/验证误差；本组拟合实验按训练误差选检查点。GroupNorm、温度退火和固定 BN 统计只保留为可选实验设置，不改变核心网络默认结构。已撤回 skip connection 和 decoder 替换。
本轮推荐权重为 `checkpoints/corner_predictor_fixed.pt`，`corner_predictor_paper.pt` 是相同内容的当前别名；对应元数据保存在同名 JSON。原 T=1 权重另存为 `corner_predictor_paper_T1.pt`，复查旧结果时指定 `--temperature 1`。当前结果导航位于本地 `results/decoder_fix_20261025/README.md`。

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
按顺序执行完整调试流程：
```powershell
python -m predict_corner.tools.run_experiments
```
流程为 ResNet 三组学习率对照、最佳 ResNet 的 24 帧裁剪图与完整视频测试、同学习率的 ViT 对照及视频测试。配置、源码 SHA256、阶段起止时间和控制台日志写入该次 `results/paper_supervision_时间/`，曲线分别展示 total、coarse、fine 与验证角点误差。
评估、图片预测和视频测试默认读取新训练的 `corner_predictor_paper.pt`；该权重需要先运行训练生成。可用 `--weights` 指定其他权重，ViT 同时指定 `--architecture vit`。各工具参数可用 `--help` 查看。
历史实验与视频记录见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。三个历史权重放在 `checkpoints/`，其训练和角点解码方式对应旧版，不能将其直接视为完整监督的结果。
视频 ROI 已改为 145 个机身标注关键帧及带边距的 4:3 裁剪，图片裁剪和整段视频共用相同计算。重新生成裁剪使用 `python -m predict_corner.data_prep.prepare_video_crops`；ROI 预览使用 `python -m predict_corner.tools.check_video_rois`。完整视频默认输出到 `results/roi_fixed_full_video/`，详见 [ROI_FIX.md](docs/ROI_FIX.md)。
扩充数据的训练入口为 `python -m predict_corner.tools.train_synthetic --output results/new_synthetic_training`，使用 1000 张训练图和 200 张验证图；`RenderedCornerDataset` 读取 RGB 与坐标，密集热图在 GPU 上按相同公式生成。配置、增强内容与训练前后结果见 [SYNTHETIC_V2.md](docs/SYNTHETIC_V2.md)。

## 参考
- [torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [BoxDreamer §3.3](https://arxiv.org/html/2504.07955v2#S3.SS3)
