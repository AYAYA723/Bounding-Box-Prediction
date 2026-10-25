# 视频 ROI 修正记录
2026-10-25：保留原来的 ResNet-18、loss、T=0.25 和已选权重，仅修正测试视频的裁剪区域。
- 原来 24 个时间点的框偏向手腕，快速运动时插值还会进一步偏离相机。
- 根据相机机身重新标注，共 145 个时间点；一般间隔不超过 30 帧，快速运动处进一步加密，包含视频首尾。
- 原始标注保存于 `data_prep/video_roi_annotations.json`。坐标基于 640×486 预览，使用时分别按原视频宽高缩放。
- 在机身框周围留边距，最小裁剪宽度为预览中的 160 px，保持 4:3；靠近边缘时平移裁剪框使其留在画面内。
- `prepare_video_crops.py` 和 `tools/test_video.py` 共用 `camera_rois`，24 帧裁剪与整段视频采用相同计算。
- 整段 2850 帧通过框边界、正面积和宽高比检查；另输出 287 个时间位置、574 张左右裁剪预览，复核快速转腕片段及其邻近帧。
- 视频末尾右侧机身超出原画面，输出保留其可见部分。手腕、支架造成的遮挡仍存在。
ROI 只确定网络输入区域，不是预测框或八角点真值；本轮视频仍使用原权重，不作为重新训练后的效果。
## 结果
- `results/roi_fix_20261025/roi_only.mp4`：只显示裁剪框。
- `results/roi_fixed_full_video/full_video_overlay.mp4`：新裁剪下的原模型角点预测。
- `results/roi_fixed_full_video/camera_crops.mp4`：左右裁剪图与预测。
- `results/roi_fix_20261025/manual_rois_before.json`：修正前的 24 帧 ROI 备份。
```powershell
python -m predict_corner.data_prep.prepare_video_crops
python -m predict_corner.tools.check_video_rois
python -m predict_corner.tools.test_video --device cuda --output results/roi_fixed_full_video
```
