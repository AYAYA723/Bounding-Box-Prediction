# 八角点热图标签
依据 [BoxDreamer §3.1](https://arxiv.org/html/2504.07955v2#S3.SS1)，将每张渲染图的八个二维角点转换为八通道热图。输入为 `keypoints_2d.json` 和 `rgb/`，输出为每张图片一个压缩 `.npz` 文件。当前保留原始 640×480 分辨率。

## 公式与实现选择
第 i 个角点为 `(x_i,y_i)`，中心 `c` 取八个二维角点的均值，每个通道分别计算平滑尺度：
```text
c = mean(corners_xy, axis=0)
d_i = ||(x_i,y_i) - c||_2
2*sigma_i^2 = (d_i / 10)^2
H[i,y,x] = exp(-sqrt((x-x_i)^2 + (y-y_i)^2) / (d_i/10)^2)
```
注意指数分子是距离，不是距离平方。这是论文重新定义的函数，不直接套用普通 Gaussian。这里也没有把 `sigma` 直接设为 `d_i/10`，以免分母多出一个 2。
论文仅称其为物体二维中心；八点均值的具体选择与[官方 `make_bbox_features(type="heatmap")`](https://github.com/zju3dv/BoxDreamer/blob/main/src/datasets/utils/base/bbox_utils.py) 一致。八点均值通常不等于三维中心投影，也不等于二维外接矩形中心。
官方代码在指数函数后还做了最大值归一化，并变换至 `[-1,1]`。这里按论文公式和 §3.2 的 Sigmoid 描述保留 `[0,1]`，不做这两项后处理，因此不宣称逐值复刻官方函数。论文中热图写为 `H×W×8`；文件使用适合 PyTorch 的 `8×H×W`，仅轴顺序不同。

## 坐标与标签
- `x` 是列，`y` 是行，左上像素中心为 `(0,0)`，直接使用现有投影坐标。
- 固定通道 0–7，与 `keypoints_3d.json` 的角点编号一一对应。不能根据屏幕上的左右上下重新排序。
- 保留浮点角点位置，不取整。离散网格上的最大值可能小于 1，峰值距真实角点通常不超过约 0.707 像素；这是采样误差。
- 被遮挡的虚拟包围盒角点同样生成标签。`in_image` 只表示投影是否位于画面内，不表示物体表面可见性。
- 出画角点保留原坐标，在画面范围内采样其热图尾部，不强行挪到边界；后续角点坐标损失应结合 `in_image` 处理。当前 20 张没有出画角点。
- 训练时若裁剪、缩放或翻转 RGB，要同步变换角点并重新生成热图。不要只变换图片，或把原分辨率的热图直接当作另一分辨率的监督。

## 运行
在项目根目录运行，使用现有 S 盘环境中的 NumPy 和 Pillow。PowerShell：
```powershell
& 'S:\Deep Learning\eecs498\.venv\Scripts\python.exe' -B `
  -m predict_corner.data_prep.generate_corner_heatmaps
```
默认输出 `datasets/action4/train_pbr/000000/heatmaps_boxdreamer/`。已存在时拒绝覆盖；如需生成另一版，使用 `--output '新的输出目录'`。其他场景用 `--scene '场景目录'`。
```text
heatmaps_boxdreamer/
├── 000000.npz ... 000019.npz
├── manifest.json
└── checks/
    ├── 000000_eight_channels.jpg
    ├── peak_alignment_contact_sheet.jpg
    └── validation.json
```
八通道预览每格只叠加一个通道，颜色强度直接来自标签值；白圈标记原角点，黑十字标记该通道的离散峰值。对齐总览中，彩圈是原角点，白十字是热图峰值。

## 训练时读取
```python
import numpy as np
import torch

with np.load("heatmaps_boxdreamer/000000.npz", allow_pickle=False) as label:
    target = torch.from_numpy(label["heatmaps"].copy())  # float32, (8,480,640)
    corners_xy = torch.from_numpy(label["corners_xy"].copy()).float()  # (8,2)
# DataLoader 组 batch 后 target 是 (N,8,480,640)。
# 网络经 Sigmoid 的输出与 target 对齐，再计算 SmoothL1。
```
文件还包括 `center_xy`、每个通道的 `denominators`、`in_image`、`image_size_wh`、`image_id`、`obj_id`、`gt_index`，全部为非 object 的 NumPy 数组。预览 JPG 仅供查看，训练读取 `.npz` 中的浮点数。

## 验证
生成程序检查所有 160 个通道的峰值、范围和公式采样值，并读回保存的文件检查数组一致性；输入 JSON 和 RGB 以 SHA256 核对未变。`test_corner_heatmaps.py` 另检查已知数值、非方形图像中的 x/y 顺序、亚像素位置、固定通道顺序、自适应尺度和出画角点。
```powershell
& 'S:\Deep Learning\eecs498\.venv\Scripts\python.exe' -B `
  -m unittest predict_corner.tests.test_corner_heatmaps -v
```
这些检查验证标签生成正确性，不代表网络已经训练或在真实视频上达到相同精度。
