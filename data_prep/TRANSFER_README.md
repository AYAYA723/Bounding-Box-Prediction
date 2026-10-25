# 换电脑运行
先将压缩包完整解压到一个目录。运行脚本使用相对目录，不依赖原电脑的 S 盘位置。

## 包含的内容
- 普通 Python 代码：网络、Dataset、检查、训练、预测和数据准备脚本。
- `corner_prediction/corner_predictor.pt`：20 张合成图训练 100 epochs 后的权重。使用随机初始化 ResNet-18，SmoothL1 热图损失和 AdamW，尚无真实视频上的定量精度结论。
- `corner_prediction/corner_predictor_tuned.pt`：本轮三组学习率中，按 4 张验证图的角点误差选出的 ResNet-18 权重。
- `corner_prediction/corner_predictor_transformer.pt`：沿用相同划分、学习率与训练轮数的 Transformer 对照权重。
- `corner_prediction/EXPERIMENTS.md` 与 `results/`：调参记录、每轮 CSV、指标、配置和预测预览。完整生成视频另放在 `Action4_full_video_results_20261025.zip`，便携包中保留原始视频。
- `datasets/action4/video_test_24/frames/`：从原视频等时间间隔抽取的 24 张原始分辨率图片，时间间隔约 3.96 秒。
- `datasets/action4/video_test_24/crops/`：左右相机的手工粗裁剪，用于单实例预测，不是自动检测结果或角点 GT。裁剪位置记录在 `manual_rois.json`，需结合原帧核对遮挡和裁剪范围。
- `datasets/action4/train_pbr/000000/rgb/` 与 JSON：20 张合成图及相机、位姿和角点标注。
- `head_left_rgb_raw.mp4`：原始视频。
- `相机资产/0468e0a404ddc50b1663be3cac636dd8.glb`：原始生成模型。
热图数组没有放入压缩包，可由二维角点在本机重新生成。环境、缓存、论文、题目、汇报和第三方仓库均不包含在包内。

## 环境
建议使用 Python 3.12 或更新版本创建独立环境；网络代码只需要 PyTorch、torchvision、NumPy 和 Pillow。在包的根目录打开终端：
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```
CPU 测试：
```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy Pillow opencv-python matplotlib
```
如果已有可用的 PyTorch 环境，直接使用现有环境即可。GPU 版本按新电脑显卡、驱动选择 PyTorch 安装命令；原电脑的 `requirements.txt` 固定的是 CUDA 13.0，不能直接据此认定新电脑兼容。

## 先看预测
CPU 可直接运行；有可用 CUDA 环境时将 `cpu` 改为 `cuda`：
```powershell
.\.venv\Scripts\python.exe corner_prediction/predict.py --device cpu --weights corner_prediction/corner_predictor_tuned.pt
```
结果保存在 `results/video_test_24/`，其中 `predictions_contact_sheet.jpg` 为总览，`predictions.json` 为预测坐标。默认输入是手工裁剪的单个目标区域。若直接输入含两台相机的完整帧，当前网络仍只输出一组八角点。
Transformer 对照：
```powershell
.\.venv\Scripts\python.exe corner_prediction/predict.py --device cpu --architecture vit --weights corner_prediction/corner_predictor_transformer.pt --output results/transformer_crops
```
完整视频测试（CPU 会比 GPU 慢）：
```powershell
.\.venv\Scripts\python.exe corner_prediction/test_video.py --device cpu
.\.venv\Scripts\python.exe corner_prediction/test_video.py --device cpu --architecture vit --weights corner_prediction/corner_predictor_transformer.pt --output results/transformer_full_video
```
完整视频的目标区域由 24 组手工裁剪框线性插值得到，画面会明确标注。这不是自动检测器，也没有真实角点 GT。
查看合成图上的预测：
```powershell
.\.venv\Scripts\python.exe corner_prediction/predict.py --device cpu --input datasets/action4/train_pbr/000000/rgb --output results/synthetic_preview
```

## 重新检查或训练
先生成省略的热图标签：
```powershell
.\.venv\Scripts\python.exe data_prep/generate_corner_heatmaps.py
```
再运行 `corner_prediction/check_model.py` 或 `corner_prediction/train.py`。这两个脚本当前使用 CUDA，在 CPU 上运行时把脚本中的 `torch.device("cuda")` 改为 `torch.device("cpu")`。`train.py` 会从随机初始化开始，训练后覆盖同名权重；需保留当前版本时先复制 `corner_predictor.pt`。

## 接下来做什么
1. 先检查 20 张合成图上的角点叠加，统计二维角点像素误差，确认模型是否真正拟合训练集。只看热图 loss 下降不够。
2. 再看 24 帧裁剪图上的输出，记录偏移、角点聚集、遮挡和背景干扰；这些视频帧继续只作为测试数据，不用于训练。
3. 若合成图本身仍不准，先改善角点定位监督，接入可微坐标提取与 fine loss，再考虑预训练 ResNet 特征或更长训练。
4. 能拟合小样本后，再按 HCCEPose 原脚本要求扩充渲染数据，增加姿态、尺度、真实背景、光照及手臂/支架遮挡。
5. 最后接目标检测/裁剪和 PnP，处理双实例，并确认拍摄相机的真实内参。原渲染 K 不能直接当作视频相机 K。
下一次汇报先展示“合成数据 → 角点预测 → 视频初测”的闭环、失败例和下一步实验，不把流程跑通等同于精度达标。
