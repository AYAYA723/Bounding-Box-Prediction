# Action 4 数据准备
当前模型来自腾讯混元生成结果 `../../相机资产/0468e0a404ddc50b1663be3cac636dd8.glb`，原文件未修改。HCCEPose 仓库位于 `../../HCCEPose`，本次对应 commit `f75cc4d5db2fc298fc7b853f8c75273757872382`。

## 当前内容
- `models/obj_000001.ply`：BOP ASCII PLY，毫米单位，含顶点法线、UV 和三角面。
- `models/obj_000001.png`：从原 GLB 提取的 4096×4096 Base Color 贴图，文件内容不变。
- `models/models_info.json`：物体 1 的包围盒、各轴尺寸与模型直径。
- `camera.json`：仓库示例的 640×480 渲染相机参数，**不是目标视频的实测标定值**。
- `keypoints_3d.json`：包围盒八角点坐标、编号及连线关系。包围盒包含凸出的镜头；角点是虚拟长方体角点，不是网格表面上的八个特征点。
- `metadata/preparation.json`：源文件与输出文件 SHA256、坐标变换、处理参数和验证结果。
- `metadata/camera_provenance.json`：示例相机参数的来源与用途。
- `checks/`：PLY 重新导入后的四视角图片、角点叠加图、投影检查记录。

## 尺寸和坐标
先按宽 70.5 mm 等比缩放，保留生成模型的形状，没有沿各轴拉伸。当前模型宽×高×厚约为 **70.5×44.5987×31.3455 mm**；与[官方规格](https://www.dji.com/osmo-action-4/specs) 70.5×44.2×32.8 mm 仍有偏差。模型直径约 80.99997 mm，按网格凸包顶点的最远点对计算，不是包围盒对角线。

坐标原点位于完整模型的轴对齐包围盒中心。面向相机正面时，X 向右、Z 向上，Y 指向机身背面，前表面法向为 −Y。八角点 0–3 是底面，4–7 是顶面，对应角点按 Z 方向连接；完整坐标和前表面四点见 `keypoints_3d.json`。

只做必要的数据清理：移除 1 个零面积三角面，修复 34 个使用中顶点的零法线。当前为 881218 个顶点、1496631 个三角面，未减面。

PLY 保留 Base Color 贴图和几何顶点法线，不完整保存 GLB 的 normal map、metallic/roughness 材质。这些原始 PBR 数据仍在未修改的源 GLB 中。

## 已验证和待做
已用仓库 `bop_toolkit.zip` 内的原始 `load_ply` 函数读取模型，并检查贴图、法线、UV、毫米尺度、包围盒中心和角点一致性。随后在独立 Blender 5.2 进程重新导入 PLY，按仓库加载器的 UV 字段映射恢复贴图，检查四个透视视角的角点投影。

`checks/` 是数据准备检查图，**不是已经完成的训练数据集**。完整 HCCEPose/BlenderProc 渲染环境尚未单独配置，尚无训练结果。

## 20 张带标签小样本（已完成）
位置：`train_pbr/000000/`。640×480，Seed 42，覆盖正面、背面、侧面与不同俯仰角，并随机化滚转、距离、灯光和纯色背景。只渲染一个无外部遮挡的相机实例，没有使用目标视频帧。

本批由 `data_prep/render_action4_smoke.py` 在已有 Blender 5.2 中生成，是兼容 BOP 的小样本适配，不是已跑通原版 HCCEPose BlenderProc 全环境。RGB 来自 Cycles，沿光轴的 Z 深度由同一网格的 BVH 在像素中心独立求交生成。深度保存为 16-bit PNG，乘以每帧 `depth_scale=0.1` 得到毫米；背景值为 0。

```text
train_pbr/000000/
├── rgb/                   20 张 RGB 图片
├── depth/                 20 张深度图
├── mask/                  20 张完整物体掩膜
├── mask_visib/            20 张可见掩膜（本批与完整掩膜相同）
├── scene_camera.json      每帧 K、depth_scale
├── scene_gt.json          每帧 obj_id、R_m2c、t_m2c（毫米）
├── scene_gt_info.json     轮廓包围框、像素数量与可见比例
├── keypoints_2d.json      每帧固定编号的八角点投影
├── run_manifest.json      参数、来源、限制、验证摘要与输出 SHA256
└── checks/                原始 RGBA、深度浮点值、叠加图、汇总图与 validation.json
```

20 张均通过检查：相机投影最大差异 `0.0000666 px`；BVH 掩膜与独立 Cycles alpha 轮廓最低 IoU `0.999285`；PnP 回代、位姿单位、深度编码和文件数量检查均通过。这些指标只说明合成标签内部一致，**不是训练后模型的预测精度**。八角点包括被遮住的虚拟包围盒角点，`in_image` 仅表示是否在画面内，不表示可见表面。

优先查看 `train_pbr/000000/checks/labels_contact_sheet.jpg` 和 `mask_alignment_worst4.jpg`。当前还没有 HCCEPose 所需的正/背面稠密 XYZ 标签，后续若训练其 BF 网络，需要再生成这一部分。

下一步先让训练程序读入这批数据并做小规模验证，再扩大数据量。首周目标是最小闭环，模型精修、减面、大规模域随机化与精度调优放在流程验证之后。目标视频先仅用于测试，不从中截帧训练；若后续需要使用，应单独明确数据划分。

## 八角点热图标签（已生成）
`train_pbr/000000/heatmaps_boxdreamer/` 已有 20 份 `.npz`，每份包含 `float32 (8,480,640)` 的 `heatmaps` 及原始角点坐标。采用 BoxDreamer §3.1 的距离指数函数，中心取八个二维角点的均值，平滑分母为角点到中心距离的十分之一的平方。保留论文描述的 `[0,1]` 值域；官方代码额外的最大值归一化和 `[-1,1]` 变换未采用。
全部 160 个通道的峰值均落在距对应角点最近的像素位置，最大采样距离为 `0.667803 px`；保存后数组读回检查通过，原始 RGB 与角点 JSON 未变。检查图和结果位于热图目录的 `checks/`，生成方式、读取示例和坐标约定见 [heatmaps_README.md](../../data_prep/heatmaps_README.md)。后续改变输入图像的裁剪或分辨率时，须同步变换角点并重新生成热图。
