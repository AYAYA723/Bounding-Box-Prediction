# 扩充合成数据与训练记录
2026-10-25。沿用 ResNet-18、五级转置卷积 decoder、BatchNorm、T=0.25，以及 coarse + 2×fine loss。对比训练前后均使用修正后的 145 个 ROI 关键帧。
## 数据
数据位于 `datasets/action4/synthetic_v2/`，原有 20 张数据保留。
- `train/`：1000 张，随机种子 20261027。
- `val/`：200 张，随机种子 20261028；两组没有重复 RGB 图片。
- `pilot/`：24 张预览，用于先检查投影、热图与增强效果，不纳入正式训练。
使用已验证的自生成混元 GLB：`0468e0a404ddc50b1663be3cac636dd8.glb`，保留原始 PBR 材质。脚本沿用 HCCEPose/BOP 的坐标约定，是 Blender RGB 适配流程，未生成深度数据。
方位角覆盖约 0–360°，仰角约 −75–75°，roll 随机；改变距离和画面位置，机身包围盒投影最大边约 94–379 px。所有八个角点保留在图像内，包括被遮挡的点，沿用原来的编号和三维包围盒。
训练集含 600 张局部遮挡图、400 张模糊图；两类存在重叠。遮挡覆盖约 5%–40% 的可见机身像素，使用肤色、绑带色、布料色的程序条带，尚未使用真实手臂模型。模糊包括 Gaussian blur 和长度为 3、5、7、9 的水平或垂直 motion blur。另加入噪声、亮度和对比度变化。
光照同时在渲染和后处理阶段变化：两盏面光源的位置、能量、颜色和面积随机，环境强度为 0.25–0.85；曝光约 −0.65 至 +0.55 EV，并加入 gamma 和冷暖白平衡。
背景由独立的 HCCEPose 示例照片，以及程序生成的网格桌面、布料、木纹和墙面组成。训练/验证使用不同照片文件。没有使用目标测试视频的截帧。
RGB、完整/可见 mask、相机参数、物体位姿和八角点坐标均已保存。训练时在 GPU 上按原论文公式生成八通道热图，避免存储和反复解压约 12 GB 的密集标签；预览集另保存了 float32 NPZ。
## 标签检查
- 1200 张图均通过投影、机身 mask 范围、八个角点的图像边界和热图峰值检查。
- Blender 独立投影与保存标签的最大差异约 0.000132 px。
- 热图峰值与连续坐标的最大距离约 0.701 px，来自像素网格取整。
- NumPy 原公式与 GPU 批量公式的预览集最大差异约 5.8e-6。
- 混合精度训练时坐标解码保持 float32，避免 BF16 无法精确表示较大的像素坐标；原 float32/float64 路径保留。
- 19 项测试通过，包括原有 loss 梯度检查、公式对照和低精度坐标检查。
完整生成校验见本地 `datasets/action4/synthetic_v2/validation.json`；生成脚本快照也保存在数据目录。
## 训练
从原推荐权重继续训练，初始化 SHA256 为 `ad33e02a5db160c63fc31b7f892e0da0b129f3d1775fb3574ef547574fd0cbc6`。
配置为 batch size=8、AdamW、learning rate=1e-4、cosine decay 至 1e-6、最大 40 epochs、验证误差连续 10 轮未改善时结束。随机种子 42，训练采用 BF16，验证采用 float32。
实际完成 36 epochs，约 302 秒；最佳权重来自第 26 轮。结果保存在 `results/synthetic_v2_training/`。
| 检查项 | 训练前 | 训练后 |
|---|---:|---:|
| 新合成验证集，200 张，平均角点误差 | 127.14 px | 47.78 px |
| 其中无遮挡，80 张 | 125.00 px | 40.58 px |
| 其中有遮挡，120 张 | 128.56 px | 52.57 px |
| 新验证集角点误差 ≤10 px 的比例 | 0.56% | 13.56% |
| 原来 4 张留出图的平均角点误差 | 55.32 px | 31.92 px |
最佳权重在训练集上的 eval 平均误差为 8.55 px，明显低于验证集的 47.78 px，泛化差距仍然较大。扩充数据带来了改善，但尚不能认为已经达到稳定、精确的角点预测。
新验证集的遮挡、模糊和曝光分组不是单因素控制实验，各组也包含其他随机变化，不能将组间差异全部归因于单一增强。
## 复现与结果
在项目根目录运行：
```powershell
S:/Blender/blender.exe --background --python-exit-code 1 --python predict_corner/data_prep/render_action4_varied.py -- --output datasets/action4/new_train --count 1000 --seed 20261027 --device OPTIX
python -m predict_corner.data_prep.finish_action4_varied --scene datasets/action4/new_train --split train
python -m predict_corner.tools.train_synthetic --train datasets/action4/new_train --validation datasets/action4/synthetic_v2/val --output results/new_synthetic_training
python -m predict_corner.tools.test_video --weights results/synthetic_v2_training/best.pt --output results/synthetic_v2_video
```
生成目录需要使用新的路径。正式记录包括 config、逐轮三项 loss、训练误差、验证误差、最佳/末轮权重、可恢复的训练状态，以及训练曲线和 GT/预测叠加图。
完整视频为 `results/synthetic_v2_video/full_video_overlay.mp4`；裁剪视频为同目录的 `camera_crops.mp4`。测试视频没有角点真值，只作视觉对比。
前后同步对比为 `results/synthetic_v2_video/before_after.mp4`，左侧为原权重，右侧为扩充数据训练后的权重；三份视频均为 2850 帧，首尾解码检查通过。部分画面的预测更贴近机身，但角点对应和包围盒形状仍不稳定。
新权重另存为 `predict_corner/checkpoints/corner_predictor_synthetic_v2.pt`，同名 JSON 记录数据规模、最佳轮次、验证误差与 SHA256。训练输出目录包含生成和训练日志、源代码快照；旧推荐权重继续保留用于对照。
