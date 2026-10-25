# BoxDreamer 论文监督对齐
按用户选择，以论文的完整监督为目标：`L = L_coarse + 2 L_fine`。当前任务仍为单张 RGB 输入，使用现有 ResNet-18 或小型 ViT；此处对齐的是热图与角点监督。

## 论文与公开代码的区别
核对版本为官方仓库 commit `106f3c16c746770cbad80720ce4a557eb31f30f5`。
- [论文 §3.3](https://arxiv.org/html/2504.07955v2#S3.SS3)：coarse 和 fine 都使用 SmoothL1，fine 权重为 2。
- [官方 loss/default.yaml](https://github.com/zju3dv/BoxDreamer/blob/106f3c16c746770cbad80720ce4a557eb31f30f5/configs/model/loss/default.yaml)：fine 项被注释，默认只启用 coarse。
- [官方 recover_bb8_corners](https://github.com/zju3dv/BoxDreamer/blob/106f3c16c746770cbad80720ce4a557eb31f30f5/src/models/utils/box_utils.py)：热图最高的 20 个位置取坐标均值，仅使用 topk 的整数索引，坐标不携带梯度。执行原函数确认了 `requires_grad=False`。
- 官方热图目标先按最大值归一化再映射到 `[-1,1]`，decoder 输出也为 `2*sigmoid-1`。本版按论文公式和 Sigmoid 描述，保留目标与预测的 `[0,1]`。
审计记录保存在 `results/paper_supervision_implementation/source_audit.json`，包含固定 commit、源码 SHA256 和实际梯度属性。原仓库保存在本地缓存，未加入本项目 Git。

## 本版完整计算
`model(x)` 返回 `(heatmaps,corners_xy)`，分别为 `(N,8,H,W)` 和 `(N,8,2)`。Dataset 返回 `(image,target_heatmaps,target_corners_xy)`，GT 坐标来自已有 `.npz` 的浮点八角点，编号不变。
对每个通道的 pre-Sigmoid logits `A`，先做空间 softmax，再求像素坐标期望：
```text
P[y,x] = exp(A[y,x]) / sum_yx exp(A[y,x])
x_hat = sum_yx x * P[y,x]
y_hat = sum_yx y * P[y,x]
```
该 soft-argmax 使用温度 1，无可学习参数；参考 [Kornia 的 spatial soft-argmax](https://kornia.readthedocs.io/en/latest/geometry.subpix.html)。论文没有给出可微坐标提取的算子细节，这一步是为实现完整 fine 梯度补充的实现，不标作官方逐行复现。softmax 作用在 logits 上，热图监督仍作用在 Sigmoid 后的值上。
两个 loss 使用 PyTorch SmoothL1 默认 `beta=1`、`reduction="mean"`：
```text
coarse = SmoothL1(pred_heatmaps, target_heatmaps)
normalize(x,y) = 2 * (x/W,y/H) - 1
fine = SmoothL1(normalize(pred_xy), normalize(target_xy))
total = coarse + 2.0 * fine
```
坐标量纲参考官方角点恢复函数的 `[W,H]` 归一化约定。当前图像为非方形，明确用宽度归一化 x、高度归一化 y；不把像素坐标直接与归一化坐标混算。训练、图像评估与视频推理共用同一 soft-argmax，推理路径没有换回 argmax/topk。

## 实现检查
`test_supervision.py` 的 8 项测试通过，覆盖非方形坐标、已知峰值、loss 数值与权重、fine 独立梯度及有限差分梯度校验。ResNet 与 ViT 的真实图像前向/反向均通过，fine 单独产生的 encoder 梯度范数分别约为 `0.01157`、`0.00620`。
另对一张合成图执行 200 次更新：训练模式角点均值误差由 `122.65 px` 降到 `20.37 px`，eval 模式最终为 `25.17 px`。这是实现检查，单图尚未精确拟合，不作为验证集或视频精度。各步 total/coarse/fine 和像素误差保存为 `single_image_history.csv`。

## 运行与旧结果
```powershell
python -m unittest predict_corner.tests.test_supervision -v
python -m predict_corner.tools.check_model
python -m predict_corner.tools.verify_paper_implementation
python -m predict_corner.train
```
`train.py` 每次建立新的带时间目录，保存配置和每轮三项 loss；最终权重为 `corner_predictor_paper.pt`。调参入口 `tune.py` 同样计算完整监督并记录三项 train/validation loss，默认结果目录为 `results/paper_supervision_20261025`。
原来 coarse-only 的权重、日志、视频和便携包保留；旧视频与旧误差值没有被改写，不代表本版完整监督。新的训练与解码策略必须重新计算指标；不要把旧 argmax 报告直接与新的 soft-argmax 报告作 fine-loss 消融结论。
