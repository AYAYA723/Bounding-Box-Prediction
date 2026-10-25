# 换电脑运行
完整解压后，在包的根目录打开终端。所有代码位于 `predict_corner/`，数据位于 `datasets/`，实验记录位于 `results/`。

## 包含内容
- 网络、Dataset、loss、训练、预测、调参、单测和数据准备代码。
- `predict_corner/checkpoints/`：本地已保存的权重。三个历史权重 `corner_predictor.pt`、`corner_predictor_tuned.pt`、`corner_predictor_transformer.pt` 来自 coarse-only 训练。
- `corner_predictor_fixed.pt`：当前原架构、T=0.25 的预训练加外观增强候选，`corner_predictor_paper.pt` 为其当前别名，同名 JSON 记录配置。旧完整监督 T=1 权重备份为 `corner_predictor_paper_T1.pt`，复查时使用 `--temperature 1`。
- `predict_corner/docs/` 与 `results/`：实验条件、逐轮日志、指标和预览。完整生成视频单独打包。
- 20 张合成 RGB 与 JSON 标注；24 张视频原帧、48 张手工粗裁剪及裁剪位置记录。
- `head_left_rgb_raw.mp4` 与 `相机资产/0468e0a404ddc50b1663be3cac636dd8.glb`：原始视频与生成模型。
热图数组省略，使用已有二维角点重新生成。环境、缓存、论文、题目、汇报和第三方仓库不包含在包内。打包脚本按时间生成新文件，原有便携包保留。

## 环境
CPU 测试可使用独立环境：
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy Pillow opencv-python matplotlib
```
已有可用 PyTorch 环境时直接使用。GPU 版根据新电脑的显卡与驱动安装；项目 `requirements.txt` 对应原电脑 CUDA 13.0 环境。

## 检查与训练
下面的 `python` 指已安装依赖的解释器：
```powershell
python -m predict_corner.data_prep.generate_corner_heatmaps
python -m unittest discover -s predict_corner/tests -v
python -m predict_corner.tools.check_model
python -m predict_corner.train
```
`check_model.py` 和 `train.py` 默认使用 CUDA，CPU 运行时将其中的 `torch.device("cuda")` 改为 `torch.device("cpu")`。训练生成 `predict_corner/checkpoints/corner_predictor_paper.pt`，每次运行的配置与日志单独保存在 `results/paper_train/时间/`。

## 图片与视频预测
完成当前版本训练后：
```powershell
python -m predict_corner.tools.predict --device cpu
python -m predict_corner.tools.test_video --device cpu
python -m predict_corner.tools.predict --device cpu --input datasets/action4/train_pbr/000000/rgb --output results/paper_synthetic_preview
```
默认读取 `corner_predictor_paper.pt`。历史权重也可用 `--weights predict_corner/checkpoints/corner_predictor_tuned.pt` 加载；ViT 使用 `--architecture vit --weights predict_corner/checkpoints/corner_predictor_transformer.pt`。当前解码为 soft-argmax，重新预测旧权重不等于复现原来的 argmax 指标。历史结果见 `predict_corner/docs/EXPERIMENTS.md`。
默认图片输入为手工粗裁剪的单个目标区域。完整视频使用 24 组粗裁剪框插值定位两台相机；当前没有自动检测器，也没有视频角点 GT。

## 后续实验
先检查少量合成图能否拟合，再做完整监督的学习率和架构对照。确认角点误差降低后扩充姿态、尺度、背景与遮挡，最后接入自动检测和 PnP。视频继续用于测试；PnP 需要拍摄相机内参，渲染示例 K 不能直接代用。
