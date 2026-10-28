# VGGT-Omega 官方 loss 固定副本

来源：https://github.com/facebookresearch/vggt-omega/tree/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/training

提交：`48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6`。保留原版权与 LICENSE。
Python 文件仅将 `losses.*`、`train_utils.*` 导入改为本包绝对路径，算法未改动。
`defaults.yaml` 来自该提交的 `training/config/default.yaml:loss`，仅将 `track` 设为 `null`。
FastWAM 张量适配、FP32 边界、空批次处理在上一级 `vggt_official_loss_adapter.py`。
底层相机编码/旋转函数使用仓库既有 `vggt_omega.utils`。
