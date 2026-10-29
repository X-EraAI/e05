# 发布状态

## 公开资源

代码、模型权重和数据分别发布。无需代码或模型压缩包：

- 代码：https://github.com/X-EraAI/e05
- RoboTwin 109k：https://huggingface.co/X-EraAI/e05-robotwin
- 具身预训练 VGGT：https://huggingface.co/X-EraAI/e05-embodied-vggt
- 处理后数据：https://huggingface.co/datasets/X-EraAI/e05-robotwin-data

固定版本、模型下载及加载检查、全量数据下载和安装验证见
[Hugging Face 发布记录](validation/huggingface_release.json)。
模型摘要及资源用途见 [checkpoint 清单](../configs/resources/checkpoint_manifest.json)。

## 已完成的本地验证

| 检查 | 结果 | 保留的记录 |
|---|---|---|
| 原始 109k 权重 | 文件内容及 SHA256 保持不变 | [权重校验](validation/release_checkpoint_after_smoke.json) |
| 单机 8 卡训练 smoke | 每卡 batch 8、累积 2，4 次 optimizer update；公开 VGGT 初始化，ActionDiT/bridge 随机初始化 | [训练摘要](validation/training_smoke.json) |
| smoke 权重重载 | 动作及几何输出检查通过 | [重载摘要](validation/training_smoke_reload.json) |
| 109k 单机 8 卡评测 smoke | 16 个 rollout 正常结束，0 worker 异常；clean 7/8、random 4/8 | [评测摘要](validation/evaluation_smoke.json) |
| 回归测试 | 95 项通过 | [测试报告](validation/regression.xml) |
| Hugging Face 资源验收 | 两个模型下载及加载通过；全部处理后数据安装及覆盖校验通过 | [发布记录](validation/huggingface_release.json) |

每卡 batch 16 的两次尝试均发生 OOM，不能标为通过。
正式训练默认 batch 8、累积 1；上述 smoke 使用累积 2，有效 batch 为 128，
不等同于原始 32 卡、有效 batch 256 的完整训练复现。
历史 benchmark 保留在 `docs/validation/historical/`，与 smoke 结果分别记录。

## 仓库记录与运行输出

资源清单位于 `configs/resources/`；公开验证摘要和历史 benchmark 位于
`docs/validation/`。摘要中的旧运行路径仅用于追溯，不表示对应临时文件仍然存在。
开发过程记录不再纳入 Git，现有本地副本归档于被忽略的 `audit/artifacts-before-cleanup/`。

环境、数据检查及回归测试报告写入被 Git 忽略的 `outputs/validation/`；
训练和评测输出继续使用 `runs/`。重新运行检查不会覆盖公开的历史验证摘要。
发布权重、数据、仿真资产及独立 Conda 环境均不作为 Git 内容提交。

## 验证范围与限制

原仓库及环境的历史隔离检查记录了外部工作区/pip 变化，不能宣称当时整个源环境完全未变；
相关开发过程记录已移出代码仓库。
本地独立环境已验证可运行，从零执行锁定安装脚本的全过程未另行重跑。
许可适用范围见 [第三方声明](../THIRD_PARTY_NOTICES.md)。
