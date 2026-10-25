# 训练调优记录
2026-10-25 更新：以下历史学习率和架构对照均为 coarse-only + argmax 版本。现已按用户选择改为论文完整监督，详细实现与官方代码差异见 [BOXDREAMER_ALIGNMENT.md](BOXDREAMER_ALIGNMENT.md)。旧 coarse-only 结果目录已按要求删除，本文保留历史数值摘要；新的完整监督实验单独记录。
## 目标与评价方式
先确认模型能否学到包围盒角点，再看真实视频上的表现。合成数据使用二维角点的平均欧氏像素误差，以及误差不超过 5/10 px 的角点比例；热图 loss 单独记录。视频没有角点 GT，只保存可视化，不报告定量精度。

## 历史基线：20 张合成图
2026-10-24：随机初始化 ResNet-18，20 张图片全部用于训练，batch size=2，AdamW，learning rate=`1e-4`，100 epochs，只有 SmoothL1 热图损失。训练 loss 从 `0.054572` 降至 `0.017428`。原始权重为 `corner_predictor.pt`，每轮记录为本地 `results/baseline_training/training_history.json`。该实验没有验证集，不能把训练 loss 当成泛化结果。

## 2026-10-25：学习率对照
固定随机种子 42，将 20 张合成图划分为 16 张训练、4 张验证；所有学习率使用相同划分、初始化种子和采样顺序。batch size=2、100 epochs、AdamW、weight decay=0.01、SmoothL1 热图损失保持一致，仅对照 `1e-4`、`3e-4`、`1e-3` 三组学习率。历史 20 张基线与此处划分不同，分开报告。
每轮将 train loss 写入 CSV；第 1 轮及每 10 轮记录验证 loss 和角点误差，按验证角点误差选择 best checkpoint，同时保留 last checkpoint。具体划分在 `results/tuning_20261025/split.json`，各组保存 `config.json`、`history.csv`、`metrics.json` 和角点叠加图。
| Learning rate | 验证误差最低轮次 | 对应训练角点误差 | 验证角点误差 | 验证角点 ≤10 px |
| --- | --- | --- | --- | --- |
| `1e-4` | 30 | 132.12 px | 114.99 px | 3.125% |
| `3e-4` | 70 | 147.02 px | 110.04 px | 0% |
| `1e-3` | 80 | 187.52 px | 137.89 px | 3.125% |
本轮按验证平均像素误差选择 `3e-4` 第 70 轮，仅表示三组候选中该指标最低。三组均未达到准确定位，学习率调节本身未解决热图峰值不准的问题。4 张验证图的结论仍需更多数据检验。
历史 20 张全训练基线在 eval 模式下：heatmap loss=`0.024991`，平均角点误差=`105.90 px`，中位数=`75.42 px`，≤5 px 为 `0.625%`，≤10 px 为 `1.875%`。这里的 eval loss 与训练日志中的每轮 batch 平均 loss 分开记录。

## 后续测试顺序
先用本轮最佳 ResNet-18 跑原始视频的全部 2850 帧。目标区域通过 24 组手工粗裁剪框做时间插值，明确不算自动检测，视频无 GT，不报告角点准确率。保存整段叠加视频、左右相机放大视频及逐帧坐标。
完整视频测试结束后，再用相同划分、seed、batch size、100 epochs 和 `3e-4` 学习率进行 Transformer 对照，保持 decoder 和输出头一致。该学习率来自 ResNet 搜索，不预先认定为 Transformer 的最优值。
ResNet 完整视频已完成：2850/2850 帧、左右目标共 5700 次预测，两个输出视频均保留 2850 帧，处理耗时约 123.46 秒。结果为 `results/resnet_full_video/full_video_overlay.mp4` 和 `camera_crops.mp4`，逐帧坐标为 `predictions.jsonl`。预览中仍有明显角点错位与连线变形。

## Transformer 对照设置
在上述完整视频测试完成后启动。使用小型 ViT：32×32 patch，15×20 共 300 个 tokens，embedding dim=384，6 个 encoder blocks，6 个 attention heads，MLP dim=1536，dropout=0.1，学习位置编码，不使用 CLS token 或预训练权重。编码后用 1×1 Conv 投影到 512×15×20，decoder 和输出头与 ResNet 相同。
沿用同一 16/4 划分、seed=42、batch size=2、100 epochs、AdamW、weight decay=0.01 和 heatmap SmoothL1，仅运行学习率 `3e-4`，不另行搜索 Transformer 学习率。完整日志位于 `results/transformer_20261025/`。
100 轮训练和评估均已完成；在最后写 comparison.json 时，相对 `--output` 与项目绝对路径的转换报错。权重、CSV、metrics 和预览已保存，修正为先 resolve 输出路径后，根据已保存的实际 metrics 补写汇总，未重复训练。

| 架构 | 参数量 | 最佳轮次 | 训练平均角点误差 | 验证平均角点误差 | 验证角点 ≤10 px |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 | 13,971,608 | 70 | 147.02 px | 110.04 px | 0% |
| 小型 ViT | 14,935,000 | 100 | 122.63 px | 105.81 px | 3.125% |
本次小型 ViT 的验证均值比 ResNet 低 4.23 px，改善约 3.84%，但验证集仅 4 张/32 个角点；两版离准确定位都很远，不能据此得出 Transformer 普遍更好的结论。相同条件的对比汇总和图见 `results/architecture_comparison/`。
Transformer 也已完成全视频测试：2850/2850 帧、5700 次目标预测，原画面叠加视频与左右放大视频均为 2850 帧，耗时约 117.84 秒（包含读写与绘制，不是纯网络速度基准）。四段视频的首尾帧均可解码，逐帧 JSONL 行数均为 2850。预览中两版均存在明显的连线变形与角点错位。
相对输出路径修复后，另用 CPU 单轮训练验证完整写入流程通过；该运行位于 `.cache`，仅作程序检查，不计入上表的模型实验。

## 下一步实验
先做单张/少量合成图的过拟合检查，直接观察角点坐标误差；验证加入可微坐标提取与 fine loss 后是否能改善峰值定位。确认小样本能拟合后，再扩充姿态、背景、遮挡等合成数据，并比较使用预训练特征的效果。视频仍只用于最终检查，真实内参和双实例自动检测尚需单独完成。

## 完整监督实现检查
按论文采用 `L_coarse + 2*L_fine`，补充温度 1 的 logit soft-argmax；预测和 GT 坐标都使用 `2*xy/[W,H]-1` 后计算 SmoothL1。热图保留 `[0,1]`，不改变已有论文公式标签。
8 项数值与梯度单测通过，两个模型的 fine 独立梯度均能传回 encoder。ResNet 单图 200 步检查：train 角点误差 `122.65 → 20.37 px`，最终 eval 为 `25.17 px`。完整记录为 `results/paper_supervision_implementation/verification.json` 和 `single_image_history.csv`。
这些是实现与短训练检查，不与完整调参结果混算。

## 完整监督重新调试
2026-10-25 10:19 启动，输出为 `results/paper_supervision_20261025/`。已清理 `baseline_training`、`baseline_evaluation`、`tuning_20261025`、`transformer_20261025`、`architecture_comparison`、`resnet_full_video`、`transformer_full_video` 七个旧目录，共约 993.8 MiB；清理清单见 `results/coarse_only_cleanup_20261025.json`。保留已有 fine loss 实现检查。
重新使用固定 16/4 划分、seed=42、batch size=2、AdamW、weight decay=0.01、每组 100 epochs。ResNet 比较 `1e-4 / 3e-4 / 1e-3`，按验证角点平均像素误差选择权重；先完成最佳 ResNet 的完整视频，再以其学习率训练同预算 ViT。全程使用 `coarse + 2*fine` 和温度 1 的 logit soft-argmax。
每轮记录 train total/coarse/fine；第 1 轮及每 10 轮记录验证三项 loss、角点误差。各阶段命令、源码 SHA256、起止时间、退出状态及日志写入 `pipeline.json` 和 `logs/`。视频仍无角点 GT，仅作可视化检查。

### 本轮最终结果
运行于 10:19:41–10:28:09，所有阶段均正常退出。ResNet 三组结果如下：
| 学习率 | 最佳轮次 | 训练平均角点误差 | 验证平均角点误差 | 验证角点 ≤10 px |
| --- | --- | --- | --- | --- |
| `0.0001` | 50 | 86.43 px | 101.29 px | 3.125% |
| `0.0003` | 100 | 66.54 px | 100.28 px | 0.000% |
| `0.001` | 70 | 98.80 px | 103.10 px | 0.000% |
| 架构 | 学习率 | 最佳轮次 | 训练平均误差 | 验证平均误差 | 验证角点 ≤10 px |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 | `3e-4` | 100 | 66.54 px | 100.28 px | 0% |
| 小型 ViT | `3e-4` | 70 | 90.30 px | 111.96 px | 0% |
ResNet 最佳验证 loss：coarse=`0.029396`、fine=`0.034712`、total=`0.098821`；ViT 分别为 `0.028232`、`0.044895`、`0.118022`。fine 已实际参与全部 400 轮训练，日志恒等式检查最大误差小于 `1e-8`。
ResNet 全视频于 10:24:52 完成，随后才启动 ViT 训练。两版均完成 48 张裁剪图和全部 2850 帧测试；四段 MP4 均为 2850 帧，首尾帧可解码，JSONL 数量和八角点坐标检查通过。完整结果导航为本地 `results/paper_supervision_20261025/README.md`。
两版预测仍存在角点向内聚集、框体变形；训练误差也较大。下一步优先完成单图精确拟合检查，观察热图与坐标解码、train/eval 差异，再扩充数据。当前数据量仅为 16/4，不把此轮差异推广到架构的普遍优劣。

### 准确率排查
使用现有权重进行只读诊断，权重 SHA256 与完整调参结束时一致。完整说明及原始数值位于 `results/paper_supervision_20261025/diagnostics/`。
关键发现：正确 GT 热图经当前 T=1 logit soft-argmax，验证平均误差仍为 32.66 px；同一热图 argmax 为 0.384 px，top-20 为 0.151 px，T=0.25 为 0.213 px。当前标签热图半峰值半径中位数约 110 px，峰值与概率均值并不一致。先前检查了梯度，但缺少 GT 热图经解码还原坐标的兼容性验证。
现有 ResNet 权重直接改用 T=0.25、argmax、top-20，验证误差分别为 116.05、141.19、113.12 px，均未改善原来的 100.28 px，说明问题也包含热图学习不足。固定输出训练集平均角点的验证误差为 104.46 px，模型相对该无图像基线的收益很小。
重新投影 160 个角点与 NPZ 标签一致。仅用训练图重估 BN 后，ResNet 验证误差为 99.93 px，变化不足以解释整体问题。下一步先验证标签与可微解码的一致性，再做单图精确拟合，随后检查预训练和数据扩充；本次没有更改模型或重训。

## 原架构的小改动修复
按最新要求保留 ResNet/ViT、原转置卷积 decoder 和默认 BatchNorm，撤回 skip connection 与 decoder 替换。GroupNorm 仅作为 `tools/fit.py` 的可选实验，核心网络中没有引入该改动。
最终默认 soft-argmax 温度为 0.25。GT 回代平均 0.201 px、最大 0.286 px；单图 1700 步后平均 0.966 px、最大 1.491 px。两项检查通过。
三组原架构对照保持 16/4 划分、seed=42、batch size=2、5000 步、AdamW 3e-4、余弦衰减到 1e-6、T=0.25，按训练误差选检查点：
| 设置 | 更新步数 | 训练平均误差 | 验证平均误差 |
| --- | --- | --- | --- |
| 随机初始化 | 5000 | 46.37 px | 127.79 px |
| ImageNet 预训练 | 5000 | 6.98 px | 74.44 px |
| 预训练 + 外观增强 | 5000 | 2.91 px | 55.32 px |
外观增强只替换训练图背景、改变颜色和轻微模糊，不改变角点坐标，不增加三维姿态，不使用目标视频。预训练权重下载至 S 盘 `.cache/torch/` 并校验。
固定 BN 统计后做额外拟合检查，训练平均误差 0.642 px，但最大 19.35 px，尚未通过全部角点最大误差 ≤5 px 的门槛；其验证误差为 66.33 px。对 000003.png 的短程复查达到平均 0.115 px、最大 0.284 px；000008.png 尚有 9.48 px 异常。没有把这些局部拟合成绩当作整体泛化成绩。
选用验证平均误差 55.32 px 的预训练+增强候选。16 张精确拟合和验证精度均仍有剩余问题，本轮不继续扩大网络结构改动。所有成功、失败和取消实验均保存在 `results/decoder_fix_20261025/`；结果导航为该目录的 README。

## 单图 DINOv2 + Transformer
采用 DINOv2 ViT-B/14 reg 通用预训练，冻结 encoder，随机初始化并训练角点 decoder；未加载 BoxDreamer 任务权重。decoder 为 12 层、768 维、8 heads，Linear/unpatchify 输出八通道热图。继续使用 coarse + 2×fine 和 T=0.25。
初次单图 lr=3e-4 的平均误差约 37.61 px，降到 1e-4 后约 7.81 px，存在过大 logits 和错误峰值。增加输出 LayerNorm、小幅预测头初始化后，独立复核在第 550 步达到平均 0.96 px、最大 1.62 px。
相同 1000/200 合成划分下完成首轮 20 epochs，batch size=8、AdamW lr=1e-4、cosine 至 1e-6。最佳第 20 轮：训练 eval 平均 26.57 px，验证平均 38.98 px；原 ResNet 为 8.55 / 47.78 px。新方案减少较大错误，但 ≤10 px 的角点比例从 13.56% 降到 7.63%，精度仍不均匀。
27 项测试通过。全验证集的缓存/在线坐标一致，encoder 哈希保持不变。参数、optimizer、scheduler 和随机状态均保存，续训加载后 AdamW step 正确接到 2501。架构图、运行命令、续训方式和视频路径见 [DINO_TRANSFORMER.md](DINO_TRANSFORMER.md)。
