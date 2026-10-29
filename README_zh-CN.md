# E0.5

**From Reconstructive to Future-Predictive 3D Foundations for Robotic World-Action Modeling**

**从几何重建到未来预测：面向机器人世界—动作建模的三维基础模型**

**X-Era AI Lab**

[论文](https://arxiv.org/pdf/2610.36011) · [项目主页](https://www.x-era.ai/e05/index.html)

[English](README.md) | 简体中文

本仓库是 **E0.5** 的官方实现。E0.5 以三维基础模型为核心，在统一的几何表示上进行机器人世界建模与动作预测。模型以视觉观测、语言指令和机器人状态为条件，联合预测机器人动作与未来三维几何，将对当前场景的几何重建扩展为对交互过程中世界变化的预测。

本仓库提供 E0.5 模型实现、训练与推理流程，以及 RoboTwin 评测工具。公开资源包括 RoboTwin 策略权重、处理好的训练数据，以及用于真机任务适配的具身预训练 VGGT 骨干权重。Python 包和配置入口沿用 `e05` 命名；资源需求与使用方法详见下文。

## 模型概览

| 组成 | 规格 |
|---|---|
| 视觉观测 | 三路相机，顺序为 `cam_high`、`cam_left_wrist`、`cam_right_wrist` |
| 语言输入 | 自然语言任务指令 |
| 机器人状态 | 14 维关节与夹爪状态 |
| 动作输出 | 32 × 14 的关节动作序列，恢复至数据集原始单位 |
| 辅助目标 | 相对当前帧偏移 32 帧的未来几何 |
| 训练初始化 | RoboTwin 使用公开 VGGT-Omega；真机使用 e05 具身预训练 VGGT。ActionDiT 和 bridge 随机初始化。 |

几何真值用于训练时的重建损失，不是推理输入。公开 Python 接口为：

```python
from e05 import E05, create_e05
```

Hydra 配置使用 `e05.create_e05`，RoboTwin 使用 `task=e05` 和 `e05_policy`。实现与共享组件统一位于 `src/e05/`，加载器兼容原始发布权重。详见[命名说明](docs/NAMING.md)和[模型卡](docs/MODEL_CARD.md)。

## 安装

在仓库根目录执行：

```bash
bash scripts/setup_env.sh
source scripts/env.sh
python scripts/check_env.py
python scripts/download_components.py
```

安装脚本在 `.envs/e05` 创建独立 Conda 环境。提供的环境检查面向具有 8 张可见 CUDA GPU 的 Linux 节点；独立模型推理使用单张 GPU。运行需要兼容的 NVIDIA 驱动和 CUDA Toolkit，RoboTwin 评测还需要 Vulkan 渲染支持。

依赖版本记录在 `requirements.lock.txt` 和 `environment.conda.lock.txt`。每次打开新的 shell 后，请先执行 `source scripts/env.sh`，以配置模块导入路径和项目内缓存。组件下载脚本根据[记录的摘要](configs/resources/component_manifest.json)校验公开 VGGT 初始化权重、文本编码器及 tokenizer；该脚本**不会**下载训练完成的 e05 109k checkpoint、具身预训练 VGGT checkpoint 或处理好的训练数据。

环境检查和仿真配置详见[安装说明](docs/INSTALL.md)。独立 checkout 中发现的安装问题、修复和验证范围记录在[独立复现说明](docs/INDEPENDENT_REPRODUCTION.md)。

## 数据与权重

根据需要运行的流程准备资源：

| 流程 | 所需资源 |
|---|---|
| RoboTwin 训练 | 处理好的 LeRobot 数据、重建缓存、文本 embedding 与 mask、归一化统计，以及公开 VGGT 初始化权重 |
| 真机训练 | 对应任务的处理后数据与统计、文本组件，以及 e05 具身预训练 VGGT 权重 |
| 独立推理 | 训练完成的 e05 checkpoint、对应归一化统计、文本组件，以及输入观测 |
| RoboTwin 评测 | 训练完成的 e05 checkpoint、对应归一化统计、文本组件，以及兼容的仿真资产 |

默认本地目录布局如下：

```text
data/
├── dataset_stats.json
├── robotwin_clean_xera_v21/
│   ├── data/
│   ├── meta/
│   └── videos/
├── reconstruction_cache/
│   ├── manifest.json
│   └── shards/
└── text_embeds_cache/robotwin/
    ├── manifest.json
    └── shards/
checkpoints/
├── release/
│   ├── step_109000.pt
│   └── dataset_stats.json
├── embodied-vggt/
│   └── e05_embodied_vggt_step_260000.pt
└── ...                         # 下载脚本放置的公开组件
third_party/RoboTwin/assets/    # 仿真评测所需资产
```

请使用与 checkpoint 对应的归一化统计。RoboTwin 参考训练数据包含 50 个机器人任务、2,500 个 episodes、549,787 帧，元信息标注为 15 FPS。请保留原有帧索引和缓存对应关系；本仓库不提供原始数据采集或标签重新生成流程。

参考数据准备完成后，可执行完整校验：

```bash
python scripts/validate_data.py --data-root ./data --full
```

如果数据位于其他路径，将 `E05_DATA_ROOT` 设置为包含 `dataset_stats.json` 及上述三个数据/缓存目录的根目录，并将该目录传入 `--data-root`。校验脚本检查的是参考数据集的完整覆盖关系。数据格式和来源详见[数据卡](docs/DATA_CARD.md)。

109k checkpoint 是原 70-epoch 训练预算中的中间权重，其状态记录为 epoch 50。权重标识见[模型卡](docs/MODEL_CARD.md)。开源 checkpoint 资源列于下表；运行相应流程前，需要先取得所需权重与数据。

| Checkpoint 资源 | 用途 |
|---|---|
| `release/step_109000.pt` | 完整 RoboTwin e05 模型，用于推理和评测 |
| `embodied-vggt/e05_embodied_vggt_step_260000.pt` | 用于真机训练的具身预训练 VGGT，从 step-260000 的 e05 完整模型中提取 |

上述路径均相对于 `checkpoints/`。具身预训练权重包含 aggregator、语言 projector、camera head 和 dense head，不包含 action expert、bridge 或 proprio encoder。摘要、来源和加载方式见 [checkpoint 资源清单](configs/resources/checkpoint_manifest.json)和[权重说明](docs/CHECKPOINTS.md)。权重作为独立 checkpoint 资源提供，不作为二进制文件提交至 Git；公开下载入口见下方。

### 下载已发布资源

- [RoboTwin 109k 完整策略](https://huggingface.co/X-EraAI/e05-robotwin)
- [具身预训练 VGGT](https://huggingface.co/X-EraAI/e05-embodied-vggt)
- [处理后的 RoboTwin 数据](https://huggingface.co/datasets/X-EraAI/e05-robotwin-data)

按所需流程选择下载。RoboTwin 训练仍从公开 VGGT 初始化；109k 用于推理和评测，具身预训练 VGGT 用于真机训练。处理后数据约 378 GiB；下载和安装副本并存时需要两份数据空间及解压开销。以下命令固定到已发布的 Hub 版本。

```bash
source scripts/env.sh
hf download X-EraAI/e05-robotwin --revision 886707967d97c0122e31426b6792799da1322fbd --local-dir checkpoints/release
hf download X-EraAI/e05-embodied-vggt --revision 2fdf396328d0c6808e2d8492fbd80b7f77bbe929 --local-dir checkpoints/embodied-vggt
(cd checkpoints/release && sha256sum -c SHA256SUMS)
(cd checkpoints/embodied-vggt && sha256sum -c SHA256SUMS)
hf download X-EraAI/e05-robotwin-data --repo-type dataset --revision 8c1b06a69506d2cb76dffb74529dfbc5ab894cd1 --local-dir downloads/robotwin-data
python scripts/install_data.py downloads/robotwin-data --output data
python scripts/validate_data.py --data-root data --full
```

## 训练

### RoboTwin

RoboTwin 训练从公开 VGGT-Omega 权重初始化：

```bash
bash scripts/train.sh
```

默认配置为 8 张 GPU、每卡 batch size 8、梯度累积 1、BF16 混合精度、seed 42。VGGT 与 ActionDiT、bridge 联合训练，开启图像增强及官方重建损失；重建损失权重为 `0.01`，两个分支的学习率均为 `1e-4`。W&B 默认关闭。

使用已验证的配置执行训练 smoke test：

```bash
bash scripts/smoke_train.sh batch_size=8 gradient_accumulation_steps=2
```

该命令以有效 batch size 128 完成 4 次 optimizer update，保存独立 smoke checkpoint，并重新加载以验证动作和几何推理。正式的 70-epoch scheduler 预算保持不变，仅通过 `stop_after_steps=4` 提前停止。

原始 RoboTwin 训练使用 32 张 GPU，有效 batch size 为 256。在 8 张 GPU 上执行 `bash scripts/train.sh gradient_accumulation_steps=4` 可获得相同的有效 batch 数量，但不代表精确复现原分布式训练。配置及输出说明见[训练与评测文档](docs/TRAIN_EVAL.md)。

### 真机适配

真机训练使用我们的**具身预训练 VGGT** 权重初始化 VGGT 分支，与 RoboTwin 使用公开 VGGT-Omega 初始化的方式区分。该权重从 step 260,000 的完整 e05 模型中提取，保留已学习的语言投影和几何预测 heads；ActionDiT 与 bridge 独立初始化。

```bash
export E05_EMBODIED_VGGT_CHECKPOINT=/path/to/e05_embodied_vggt_step_260000.pt
```

在真机任务的训练命令中追加 Hydra 参数 `+model_init=embodied_vggt`。[初始化配置](configs/model_init/embodied_vggt.yaml)通过 `model.vggt_backbone_pretrained_checkpoint_path` 加载该权重，并关闭公开 VGGT 和完整模型初始化。请勿将 VGGT-only 权重传给 `resume` 或 `model.pretrained_checkpoint_path`。

该配置仅切换模型初始化。真机任务仍需配置对应的数据、归一化统计、相机顺序、action/state 维度和重建目标。默认训练任务及上述 smoke 结果仍为 RoboTwin；仅应用初始化配置，不代表已定义或验证完整的真机训练流程。可从[真机任务模板](configs/task/real_robot_template.yaml)开始，并用 `scripts/build_reconstruction_cache.py` 生成重建缓存，步骤见[真机流程](docs/REAL_ROBOT.md#workflow-for-your-robot)。

真机的几何监督**只使用深度**：偏移 32 帧的未来深度目标是 Depth Anything 3 估计的米制伪标签，配合相机原厂内参；不使用深度传感器，也没有相机外参。训练时需在 `model.reconstruction_loss_config` 下把 `official.camera.weight` 和 `official.point.weight` 设为 `0.0`，以关闭相机与点图损失。official 后端会忽略旧的 `lambda_camera`/`lambda_point` 键；发布的默认配置中这两项权重均为 5.0。深度监督按样本做了场景归一化（`normalize_gt: global`），因此预测深度是场景归一化单位，而不是米制。数据布局、深度归一化、重建缓存格式，以及 AgileX Piper 与 Zerith H1 的后训练超参数见[真机配方](docs/REAL_ROBOT.md)。

## 独立推理

准备包含以下字段的 NPZ 观测文件：

| 字段 | 类型与形状 |
|---|---|
| `cam_high`、`cam_left_wrist`、`cam_right_wrist` | `uint8[H, W, 3]`，RGB 图像 |
| `state` | `float32[14]`，当前关节与夹爪状态 |
| `instruction` | Unicode 字符串标量 |

```bash
python scripts/infer.py \
  --input /path/to/observation.npz \
  --checkpoint /path/to/step_109000.pt \
  --output runs/prediction.npz \
  --reconstruction
```

将 `dataset_stats.json` 放在 checkpoint 同级目录。输出包含形状为 `[32, 14]`、使用数据集原始单位的 `action`，以及用于检查的 `normalized_action`。不需要几何输出时，可省略 `--reconstruction`。预测深度采用模型学习的场景归一化尺度，不是米制深度。真机策略推理时，将其任务配置文件或训练保存的 `config.yaml` 传给 `--config`，相机字段、state 与 action 维度即按该配置读取（见[真机推理](docs/REAL_ROBOT.md#6-inference)）。

如果参考数据已放在默认路径，可执行 `python scripts/make_example.py`，将一个观测提取至 `examples/observation.npz`。使用自己的观测进行推理时，不需要训练数据集或几何真值。

## RoboTwin 评测

将兼容的仿真资产放置在 `third_party/RoboTwin/assets/`，然后编译仿真依赖并运行评测 smoke test：

```bash
bash scripts/prepare_sim.sh
export E05_CHECKPOINT=/path/to/step_109000.pt
export E05_STATS_PATH=/path/to/dataset_stats.json
bash scripts/smoke_eval.sh
```

Smoke test 在 8 张 GPU 上调度 8 个任务，每卡最多一个 worker。每个任务分别执行 1 个 clean/seen episode 和 1 个 random/unseen episode，共 16 个 rollout。固定参数为 seed 42、动作预测长度 32、每 24 步重新规划、10 步去噪。

对全部 50 个任务、每种设置下每任务 100 个 episodes 进行评测：

```bash
bash scripts/evaluate.sh EVALUATION.eval_num_episodes=100 EVALUATION.phase=both
```

日志、解析后的配置及结果保存在 `runs/`。本次代码验收未重新执行上述全量评测。

## 许可与致谢

仓库代码遵循 [LICENSE](LICENSE)，第三方材料保留各自适用条款。本项目基于 VGGT-Omega、FastWAM、RoboTwin、cuRobo 和 Wan/T5 组件构建。VGGT-Omega 包含非商业研究条款；本仓库的代码许可证不改变模型权重、处理后数据和仿真资产的许可。请参阅[第三方声明](THIRD_PARTY_NOTICES.md)及各材料附带的许可证。