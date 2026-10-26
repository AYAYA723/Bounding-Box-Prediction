# Bounding-Box-Prediction
单张 RGB 输入，预测物体三维包围盒的八个二维角点。当前使用 DINOv2 通用预训练 encoder 和自行训练的 Transformer decoder。

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
