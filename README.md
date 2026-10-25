# Bounding-Box-Prediction-with-BoxDreamer
单张 RGB 输入，预测物体三维包围盒的八个二维角点。当前使用冻结的 DINOv2 通用预训练 encoder 和自行训练的 Transformer decoder；保留 ResNet-18、小型 ViT 作为基线。未使用 BoxDreamer 的任务权重。

核心代码、运行入口和数据约定见 [predict_corner/README.md](predict_corner/README.md)。所有命令在项目根目录执行。

## 安装
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
依赖对应当前 Python 3.14 / CUDA 13.0 环境。Blender 用于合成数据，相关脚本见 [data_prep/README.md](predict_corner/data_prep/README.md)。

## 流程
```text
RGB → ROI crop → encoder → heatmap decoder → 8 ordered corners
                                              ↓
                                  3D box + camera K → PnP → R,t → box projection
```
训练使用 `coarse + 2*fine`，坐标由可微 soft-argmax 提取。视频 ROI 来自人工机身标注插值；adaptive 模式按机身大小和另一台相机的重合情况收紧边距，不包含自动检测或跟踪。

## Git 范围
保留代码、入口 README、依赖和少量标注 JSON。`predict_corner/docs/`、汇报、结果、RGB、热图、权重、模型资产、原视频、环境和缓存均保留在本地，不上传。

视频结果仍有定位偏差和朝向翻转；PnP 求解成功不代表位姿准确。现有视频内参由 AprilTag 估计，并非独立标定结果。
