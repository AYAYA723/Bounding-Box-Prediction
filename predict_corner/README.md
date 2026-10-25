# 角点预测网络
核心代码放在本目录，数据准备、训练辅助与视频测试分别放在 `data_prep/` 和 `tools/`。

```text
predict_corner/
├── model.py                 ResNet-18 + convolution decoder
├── transformer_model.py     Small ViT + convolution decoder
├── dino_model.py            Frozen DINOv2 + heatmap decoder
├── heatmap_decoder.py       Transformer + Linear/unpatchify
├── dataset.py               RGB and corner labels
├── feature_dataset.py       Cached DINOv2 features
├── supervision.py           Targets, soft-argmax and loss
├── pose.py                  PnP and box projection
├── train.py                 Basic training loop
├── data_prep/               Rendering, labels and manual ROIs
├── tools/                   Training, evaluation and video tools
└── tests/                   Numerical and geometry checks
```

## 数据
原始 20 张数据位于 `datasets/action4/train_pbr/000000/`；扩充数据位于 `datasets/action4/synthetic_v2/{train,val}/`，各场景包含 `rgb/` 和 `keypoints_2d.json`。密集热图可在 GPU 上由坐标生成。Git 仅保留少量示例标注，训练图片、模型资产与权重需另行准备。

## 网络
ResNet-18 和小型 ViT 使用转置卷积 decoder。当前 DINOv2 方案的尺寸如下：
```text
RGB                         (N, 3, 480, 640)
Pad                         (N, 3, 490, 644)
Frozen DINOv2 ViT-B/14 reg   (N, 1610, 768)
12 Transformer blocks       (N, 1610, 768)
Linear + unpatchify         (N, 8, 490, 644)
Crop padding                (N, 8, 480, 640)
Sigmoid / soft-argmax        heatmaps / (N, 8, 2)
```
DINOv2 使用通用预训练参数，decoder 从随机初始化训练。热图 loss 与归一化角点 loss 组成 `coarse + 2*fine`，soft-argmax 温度为 0.25。

## 训练
在项目根目录使用已安装依赖的 Python 环境执行。DINOv2 源码和通用权重下载至项目内 `.cache/torch/`：
```powershell
python -m predict_corner.tools.prepare_dinov2
python -m predict_corner.tools.cache_dino_features
python -m predict_corner.tools.train_dino --epochs 20 --batch-size 8 --learning-rate 0.0001 --output results/new_dino_training
```
缓存默认读取 `synthetic_v2`，写入 `datasets/action4/dino_features_v1/`。运行前先准备好 RGB 和标签；输出使用新目录。
训练目录保存 `best.pt`、`last.pt`、`training_state.pt`、配置与逐轮日志。继续训练时恢复最后一轮模型、AdamW 和随机状态：
```powershell
python -m predict_corner.tools.train_dino --resume results/new_dino_training --epochs 20 --learning-rate 0.00003 --output results/new_dino_continuation
```
ResNet 和小型 ViT 的入口仍保留：
```powershell
python -m predict_corner.tools.check_model
python -m predict_corner.train
python -m predict_corner.tools.tune --output results/new_resnet
python -m predict_corner.tools.tune --architecture vit --learning-rates 0.0003 --output results/new_vit
```

## 视频与 PnP
源视频放在根目录 `head_left_rgb_raw.mp4`。先用 `python -m predict_corner.data_prep.extract_video_frames` 生成 24 帧索引；机身关键帧标注保存在 `data_prep/video_roi_annotations.json`。裁剪后为 640×480，预测角点先还原到原图，再使用原视频内参求 PnP。
```powershell
python -m predict_corner.tools.test_video --architecture dino --weights results/new_dino_continuation/best.pt --roi-mode adaptive --sample-only --output results/new_video_samples
python -m predict_corner.tools.test_video --architecture dino --weights results/new_dino_continuation/best.pt --roi-mode adaptive --camera datasets/action4/video_camera_estimated.json --allow-estimated-camera --output results/new_video_pnp
```
`--sample-only` 只测试固定 24 帧，省略后测试整段。`--roi-mode padded` 保留旧裁剪作为对照。输出包括整帧叠加视频、相机裁剪视频和逐帧坐标/位姿记录。

`video_camera_estimated.json` 来自视频 AprilTag，焦距约 1449 px，主点固定在中心、畸变设零。它不是独立标定，使用时需显式指定 `--allow-estimated-camera`。渲染用 `camera.json` 不能替代视频拍摄内参。

## 检查
```powershell
python -m unittest discover -s predict_corner/tests -v
python -m predict_corner.tools.check_adaptive_rois --output results/new_roi_check
```
ROI 检查基于人工机身框，不代表逐帧真实轮廓始终完整；插值可能在运动较快时漏掉机身。视频没有角点或绝对位姿真值，当前仍有定位偏差与朝向翻转。

## 参考
[BoxDreamer](https://arxiv.org/abs/2504.07955)、[DINOv2](https://github.com/facebookresearch/dinov2)。
