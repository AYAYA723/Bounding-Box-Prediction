# Action 4 数据准备脚本
脚本和数据放在 HCCEPose 仓库外，未修改其上游代码。原始相机模型也不覆盖。

- `prepare_action4.py`：通过后台 Blender 读取 GLB，等比校准宽度、居中、清理无效法线和零面积三角面，导出带 UV 的 BOP PLY、Base Color 贴图、角点和元数据。
- `validate_action4.py`：用本次 HCCEPose checkout 的 BOP `load_ply` 原函数读取 PLY，通过 NumPy/SciPy 检查模型并计算直径、生成 `models_info.json`。
- `preview_action4.py`：在 Blender 5.2 中重新导入 PLY、恢复贴图，并比较 `K(RX+t)` 与 Blender 相机投影。临时导入文件仅在 `checks/` 中生成和清理，不改输入 PLY。
- `make_action4_sheet.py`：生成八角点与包围盒叠加检查图。
- `render_action4_smoke.py`：Blender 5.2 小样本适配，固定生成 20 张，输出 Cycles RGBA、BVH 光轴深度、BOP `K/R/t` 和八角点。使用单个无外部遮挡的物体，不是原版 BlenderProc 脚本。
- `validate_smoke20.py`：将 RGBA 合成 RGB 背景，保存 16-bit 深度和掩膜、BOP 轮廓信息；通过独立渲染 alpha、PnP 回代及投影检查验证标签，生成检查图和 SHA256 清单。
- `generate_corner_heatmaps.py`：依据 BoxDreamer §3.1，将八个二维角点转换为 `[0,1]` 的八通道 float32 热图，保存 `.npz`、八通道预览和峰值对齐检查；用法与论文/开源代码的差异见 [heatmaps_README.md](heatmaps_README.md)。

## 执行环境
准备和预览使用已有 `S:\Blender\blender.exe`，不需要把 HCCEPose 的全部训练依赖装进 Blender。验证和拼图使用已有 S 盘 WSL Python 环境中的 NumPy、SciPy、Pillow，没有开始 GPU 训练或安装另一套大环境。

Blender 命令应加 `--python-exit-code 1`，避免 Python 脚本报错时 Blender 仍返回退出码 0。首次准备命令示例（PowerShell）：
```powershell
& 'S:\Blender\blender.exe' --background --factory-startup --python-exit-code 1 --threads 4 `
  --python 'S:\Deep Learning\coding_test_for_research\predict_corner\data_prep\prepare_action4.py' -- `
  --source 'S:\Deep Learning\coding_test_for_research\相机资产\0468e0a404ddc50b1663be3cac636dd8.glb' `
  --output 'S:\Deep Learning\coding_test_for_research\datasets\action4' `
  --repo 'S:\Deep Learning\coding_test_for_research\HCCEPose'
```
当前数据已生成，上述首次命令会拒绝覆盖。若重新准备另一版，请改输出目录；`--resume-own-export` 仅用于接续脚本自己中断的导出，不会覆盖内容不同的现有元数据。

`camera.json` 是示例渲染内参。模型中的 Action 4 是被拍摄物体，不能把它的镜头规格填作拍摄相机标定值。已有真实视频未被用于训练或建模。

## 20 张小样本
已完成的场景为 `datasets/action4/train_pbr/000000`。渲染命令结构（PowerShell）如下；该场景已存在，脚本会拒绝覆盖：
```powershell
& 'S:\Blender\blender.exe' --background --factory-startup --python-exit-code 1 --threads 4 `
  --python 'S:\Deep Learning\coding_test_for_research\predict_corner\data_prep\render_action4_smoke.py' -- `
  --dataset 'S:\Deep Learning\coding_test_for_research\datasets\action4' --scene-id 000000 --count 20 --seed 42
```
后处理和验证（WSL Bash，使用已有 S 盘环境）：
```bash
source '/mnt/s/Deep Learning/coding_test_for_research/pixal3d_local/env.sh'
OPENBLAS_NUM_THREADS=2 python '/mnt/s/Deep Learning/coding_test_for_research/predict_corner/data_prep/validate_smoke20.py' \
  --scene '/mnt/s/Deep Learning/coding_test_for_research/datasets/action4/train_pbr/000000'
```
脚本先验证 prepared 模型的输入 SHA256，再生成新场景；不会覆盖模型或已有场景。为了先验证标签，渲染使用 4 个 CPU 线程，未占用 CUDA 推理显存。脚本本轮运行时将 Blender 临时目录和用户资源目录指定在 `.cache/asset-prep/`，均位于 S 盘。
