# Action 4 角点预测
单张 RGB 图片输入 ResNet-18，输出八通道包围盒角点热图。网络和数据接口已搭建，下一步是小样本过拟合实验。

## 文件
- [corner_prediction/check_model.py](corner_prediction/check_model.py)：读取真实数据，检查前向、反向和角点输出格式。
- [corner_prediction/train.py](corner_prediction/train.py)：按 PyTorch 基础笔记组织的训练循环和权重保存。
- [corner_prediction/model.py](corner_prediction/model.py)：ResNet-18 encoder、上采样 decoder 和热图输出层。
- [corner_prediction/dataset.py](corner_prediction/dataset.py)：RGB 与热图标签配对。
- [data_prep/](data_prep/)：模型准备、渲染适配、角点投影和热图生成脚本。
- [datasets/action4/](datasets/action4/)：少量相机参数、三维角点及 20 张图片的位姿、二维角点标注。

## 本机运行
在 VS Code 中打开本目录，Python 解释器使用：
```text
S:\Deep Learning\eecs498\.venv\Scripts\python.exe
```
该环境已有 NumPy、PyTorch、torchvision、Pillow 和 CUDA 支持。项目的本机解释器路径配置在 `.vscode/settings.json`，不纳入 Git。已有工作区若缓存了其他解释器，通过 `Python: Select Interpreter` 选择上述路径。
在项目根目录运行：
```powershell
& 'S:\Deep Learning\eecs498\.venv\Scripts\python.exe' corner_prediction/check_model.py
```
训练入口和参数见 [corner_prediction/README.md](corner_prediction/README.md)。

## 新环境
使用 Python 3.14 创建环境，安装与本机核对一致的网络运行依赖：
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
选择 `.venv/Scripts/python.exe` 作为解释器。`requirements.txt` 对应当前 NVIDIA CUDA 13.0 版本；数据准备脚本的 Blender、SciPy、OpenCV 等环境见 [data_prep/README.md](data_prep/README.md)。脚本根据自身位置定位项目目录。

## 数据与版本管理
Git 采用白名单，仅管理代码、说明、依赖清单和少量标注 JSON。渲染 RGB、深度图、掩膜、热图、三维模型、贴图、权重、环境、缓存和检查图保留在本地，通过脚本生成或另行准备。运行前需要本地已有 `datasets/action4/train_pbr/000000/rgb/` 与 `heatmaps_boxdreamer/`。
题目、任务与问题记录、论文及译稿、汇报、视频、参考素材、Pixal3D 及其部署文件、HCCEPose 上游仓库均不纳入 Git。本地文件不会因忽略规则被删除。新增顶层目录需要在 `.gitignore` 中明确允许后才会被跟踪。
