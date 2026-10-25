# Action 4 角点预测
单张 RGB 图片输入 ResNet-18 或小型 ViT，输出八通道包围盒角点热图与八个二维坐标。当前采用论文的 `coarse + 2*fine` 监督，坐标通过可微 soft-argmax 提取。历史学习率对照、架构对照和视频测试仍保留，角点定位尚不准确。

所有代码统一放在 [predict_corner/](predict_corner/README.md)。数据集放在 `datasets/`，实验记录与输出放在 `results/`，原始视频和 GLB 资产保留在原位置。

## 运行
在本项目根目录打开终端，本机使用已有环境：
```powershell
& 'S:\Deep Learning\eecs498\.venv\Scripts\python.exe' -m predict_corner.tools.check_model
& 'S:\Deep Learning\eecs498\.venv\Scripts\python.exe' -m predict_corner.train
```
在 VS Code 中选择上述 Python 解释器；`.vscode/settings.json` 是本机配置，不纳入 Git。训练、预测和调参命令见 [predict_corner/README.md](predict_corner/README.md)。

## 新环境
使用 Python 3.14 创建环境并安装当前 NVIDIA CUDA 13.0 版本的依赖：
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
选择 `.venv/Scripts/python.exe` 作为解释器。数据准备所需的 Blender、SciPy 等环境和运行命令见 [data_prep/README.md](predict_corner/data_prep/README.md)。

## 数据与记录
- [datasets/action4/](datasets/action4/README.md)：相机参数、三维角点、20 张合成图的位姿与二维角点标注。本地训练还需要 `rgb/` 和 `heatmaps_boxdreamer/`。
- [EXPERIMENTS.md](predict_corner/docs/EXPERIMENTS.md)：训练条件、调优结果和后续实验。
- [BOXDREAMER_ALIGNMENT.md](predict_corner/docs/BOXDREAMER_ALIGNMENT.md)：论文、官方代码与本版实现的差异。
当前三个历史权重和已有视频、便携包对应旧 coarse-only 版本；完整监督版本目前只做过实现和短训练检查。

Git 仅管理代码、说明、依赖和少量标注 JSON。RGB、热图、权重、三维资产、视频、环境、缓存、题目、问题、汇报、Pixal3D 和 HCCEPose 上游仓库均留在本地，不纳入 Git。
