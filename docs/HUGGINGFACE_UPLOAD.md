# e05：Hugging Face 资源上传指南

本指南按当前 E05 文件布局编写。代码继续放在 Git 仓库；Hugging Face 用于托管独立 checkpoint 和处理后数据。不需要上传代码压缩包或模型 tar 包。本项目资源已上传到 `X-EraAI` 的三个仓库，固定版本和校验状态见 [发布记录](validation/huggingface_release.json)。以下步骤保留为后续维护指南。

## 1. 上传哪些资源

推荐创建两个 Model 仓库和一个 Dataset 仓库：

| 建议仓库名 | 类型 | 本地来源 | 体积 | 用途 |
|---|---|---|---:|---|
| `你的账号或组织/e05-robotwin` | Model | `checkpoints/release/` | 权重约 6.46 GiB | 完整 109k 策略，供 RoboTwin 推理与评测使用 |
| `你的账号或组织/e05-embodied-vggt` | Model | `checkpoints/embodied-vggt/` | 权重约 4.28 GiB | 真机训练所用的具身预训练 VGGT 初始化 |
| `你的账号或组织/e05-robotwin-data` | Dataset | `dist/data/` 中已有的数据分片及清单 | 约 377.96 GiB | 复现当前 RoboTwin 训练流程 |

如果这次只发布模型，先上传两个 Model 仓库即可。若要让其他人按当前代码复现 RoboTwin 训练，第三个数据仓库也需要发布：现有配置依赖重建缓存及文本缓存，不能只上传 RGB/action 文件。

处理后数据实际组成如下：

| 内容 | 当前文件数量 | 大小 |
|---|---:|---:|
| RGB、action/state、语言及 episode 元信息 | 50 个任务 ZIP + `metadata.zip` | 约 2.29 GiB |
| Reconstruction cache | 550 ZIP、550 配套 JSON、1 个 manifest | 约 23.65 GiB |
| Text embeddings 与 masks | 256 ZIP、256 配套 JSON、1 个 manifest | 约 352.02 GiB |

还需包含 `dist/data/manifest.json`。数据清单引用的 1,665 个文件已经核对存在。`metadata.zip` 包含元信息与训练统计；两个缓存目录的 JSON 不能省略。

这里使用 `dist/data/` 只是复用现成的处理后数据分片，不要求采用之前的整套发布包。本机 `data/` 的部分缓存是指向 `dist/data/` 的相对软链接，上传结束后也应保留这些实际数据文件。

## 2. 哪些资源不必重复上传

- **代码**：在 GitHub 等代码仓库发布 `src/`、`scripts/`、`configs/`、`third_party/` 中所需源码、依赖锁文件和文档，不需要上传 `E05-code.tar.gz`。
- **公开 VGGT 初始化、文本编码器和 tokenizer**：保留 `configs/resources/component_manifest.json` 与 `scripts/download_components.py`，继续从记录的上游下载。
- **完整 step-260000 e05 源 checkpoint**：不需要上传。本次发布的是从中提取的 VGGT-only 权重，排除了 action expert、bridge 和 proprio encoder。
- **Smoke 权重、optimizer state、训练历史 runs、环境目录、缓存、账号凭据及本地 audit**：不属于这些 Hub 仓库的上传内容。
- **仿真资产**：可以引用匹配版本的 RoboTwin 上游资产。若选择另行托管，且其条款允许，可用独立 Dataset 仓库保存现有 `dist/E05-simulator-assets.zip`（约 13.95 GiB）、资产清单及来源/许可说明；不要混进模型权重仓库。

代码、权重、数据和资产的许可分别按现有 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) 及上游条款处理，不要把整个模型或数据集直接标成 MIT。

## 3. 准备 Hugging Face 账号和模型卡

先确定发布账号或组织名，以及三个仓库名。本文示例使用 `HF_NAMESPACE` 保存账号/组织名。

每个仓库需要根目录的 `README.md`：Model 仓库将其显示为模型卡，Dataset 仓库将其显示为数据卡。可以从以下文件开始编辑：

| 仓库 | 本地说明草稿 |
|---|---|
| `e05-robotwin` | `docs/MODEL_CARD.md` |
| `e05-embodied-vggt` | `checkpoints/embodied-vggt/model_card.md` |
| `e05-robotwin-data` | `docs/DATA_CARD.md` |

官方说明：[Model Cards](https://huggingface.co/docs/hub/model-cards)、[Dataset Cards](https://huggingface.co/docs/hub/datasets-cards)。

在 E05 根目录准备三份 Hub 用的说明稿：

```bash
mkdir -p audit/hf-cards
cp docs/MODEL_CARD.md audit/hf-cards/robotwin.md
cp checkpoints/embodied-vggt/model_card.md audit/hf-cards/embodied-vggt.md
cp docs/DATA_CARD.md audit/hf-cards/robotwin-data.md
```

上传前编辑这三份草稿：补充实际代码仓库链接、资源下载方式及适用许可；将代码仓库内的相对链接改成实际 GitHub 链接，移除“尚未配置下载入口”等已不适用的描述。模型卡中区分完整策略和 VGGT 初始化权重。可在 Hub 的元数据编辑器中填写 `robotics`、`e05`、`vggt` 等标签和实际许可信息，不要宣称这些自定义 `.pt` 文件可直接通过 Transformers 的 `AutoModel.from_pretrained` 使用。

数据约 378 GiB，上传前查看账号的实际存储配额。尤其不能默认免费私有空间可以容纳整个数据集；公开存储也有平台政策。参见 [Hugging Face 存储规则](https://huggingface.co/docs/hub/storage-limits)。

## 4. 使用现有 E05 环境登录

以下命令对应当前锁定的 `huggingface-hub==0.36.0`，其中命令名称为 `hf repo create` 和 `hf upload-large-folder`。不要为了发布资源升级已有训练环境。若在其他机器操作，应在独立环境安装该版本；不同版本 CLI 的命令名称可能不同。

```bash
# 在 E05 根目录执行
source scripts/env.sh
python -c 'import huggingface_hub; print(huggingface_hub.__version__)'
hf auth login
hf auth whoami
export HF_NAMESPACE=X-EraAI
```

将 `YOUR_ACCOUNT_OR_ORG` 替换为真实账号/组织名。登录时使用具有目标仓库写权限的 token；交互式输入即可。`scripts/env.sh` 将 Hugging Face 本地缓存放在 E05 内。

下面的创建命令会建立**公开仓库**。如需先保持私有，可以在相应创建命令加 `--private`，之后再从仓库 Settings 改为公开；数据仓库应先确认私有存储容量是否足够。

```bash
hf repo create "$HF_NAMESPACE/e05-robotwin" --repo-type model
hf repo create "$HF_NAMESPACE/e05-embodied-vggt" --repo-type model
hf repo create "$HF_NAMESPACE/e05-robotwin-data" --repo-type dataset
```

如果仓库已经创建，跳过对应命令。

## 5. 上传两个模型资源

### RoboTwin 109k

保持这些文件位于模型仓库根目录：

```text
step_109000.pt
config.yaml
dataset_stats.json
preprocessing.json
SHA256SUMS
README.md
THIRD_PARTY_NOTICES.md
LICENSE-VGGT-Omega
VGGT_INITIALIZATION_CARD.md
```

其中统计、相机顺序和预处理说明应与原始 109k 配套，不能换成其他任务的统计。

```bash
hf upload "$HF_NAMESPACE/e05-robotwin" checkpoints/release . --exclude model_card.md
hf upload "$HF_NAMESPACE/e05-robotwin" audit/hf-cards/robotwin.md README.md
hf upload "$HF_NAMESPACE/e05-robotwin" THIRD_PARTY_NOTICES.md THIRD_PARTY_NOTICES.md
hf upload "$HF_NAMESPACE/e05-robotwin" third_party/vggt-omega/LICENSE LICENSE-VGGT-Omega
hf upload "$HF_NAMESPACE/e05-robotwin" docs/VGGT_INITIALIZATION_CARD.md VGGT_INITIALIZATION_CARD.md
```

### 具身预训练 VGGT

模型仓库根目录包含：

```text
e05_embodied_vggt_step_260000.pt
SHA256SUMS
README.md
embodied_vggt.yaml
extraction.json
loading_validation.json
THIRD_PARTY_NOTICES.md
LICENSE-VGGT-Omega
VGGT_INITIALIZATION_CARD.md
```

```bash
hf upload "$HF_NAMESPACE/e05-embodied-vggt" checkpoints/embodied-vggt . --exclude model_card.md
hf upload "$HF_NAMESPACE/e05-embodied-vggt" audit/hf-cards/embodied-vggt.md README.md
hf upload "$HF_NAMESPACE/e05-embodied-vggt" configs/model_init/embodied_vggt.yaml embodied_vggt.yaml
hf upload "$HF_NAMESPACE/e05-embodied-vggt" docs/validation/embodied_vggt_extraction.json extraction.json
hf upload "$HF_NAMESPACE/e05-embodied-vggt" docs/validation/embodied_vggt_loading.json loading_validation.json
hf upload "$HF_NAMESPACE/e05-embodied-vggt" THIRD_PARTY_NOTICES.md THIRD_PARTY_NOTICES.md
hf upload "$HF_NAMESPACE/e05-embodied-vggt" third_party/vggt-omega/LICENSE LICENSE-VGGT-Omega
hf upload "$HF_NAMESPACE/e05-embodied-vggt" docs/VGGT_INITIALIZATION_CARD.md VGGT_INITIALIZATION_CARD.md
```

这是 VGGT 分支初始化权重，不需要搭配 RoboTwin `dataset_stats.json`；真机任务使用自己的数据统计。不要将完整 step-260000 源模型一起上传。

上述文件/目录上传方式对应 [Hugging Face 0.36.0 上传文档](https://github.com/huggingface/huggingface_hub/blob/v0.36.0/docs/source/en/guides/upload.md)。

## 6. 上传已处理的 RoboTwin 数据

保留 `dist/data/` 现有布局，将其内容放在 Dataset 仓库根目录，不额外嵌套一层 `data/` 或 `dist/`。这样用户下载后可以直接交给现有 `scripts/install_data.py`。

```bash
hf upload-large-folder "$HF_NAMESPACE/e05-robotwin-data" dist/data \
  --repo-type dataset \
  --num-workers 8 \
  --include '*.zip' '*.json'

hf upload "$HF_NAMESPACE/e05-robotwin-data" audit/hf-cards/robotwin-data.md README.md --repo-type dataset
hf upload "$HF_NAMESPACE/e05-robotwin-data" THIRD_PARTY_NOTICES.md THIRD_PARTY_NOTICES.md --repo-type dataset
```

该版本的大目录上传会保存进度，发生中断后重新执行相同命令即可继续已完成的任务。保留上传目录中的 `.cache/huggingface` 状态目录，不要同时启动多个进程上传同一个目录。权限或容量错误需要先解决原因，不能仅靠无限重试。行为依据 [0.36.0 大目录上传实现](https://github.com/huggingface/huggingface_hub/blob/v0.36.0/src/huggingface_hub/cli/upload_large_folder.py)。

上传完成后，Hub 上应有根 `manifest.json`、51 个任务/元信息 ZIP，以及两个缓存目录的 ZIP、配套 JSON 和 manifest。数据卡应列明任务数、episodes、帧数、FPS、缓存格式、来源和适用条款，不把机器人任务数与语言条目数混为一谈。

## 7. 验证下载与加载

先在独立目录下载模型，使用各自的 SHA256SUMS 核验。以下目录位于 E05 的本地缓存中，不覆盖已有发布权重：

```bash
hf download "$HF_NAMESPACE/e05-robotwin" --local-dir .cache/hf-verify/robotwin
hf download "$HF_NAMESPACE/e05-embodied-vggt" --local-dir .cache/hf-verify/embodied-vggt

(
  cd .cache/hf-verify/robotwin
  sha256sum -c SHA256SUMS
)
(
  cd .cache/hf-verify/embodied-vggt
  sha256sum -c SHA256SUMS
)
```

已准备的原始摘要：

- `step_109000.pt`：`2c6edfb1ffc2ada95c909ff69e355cdbf3a6b1abc702f77d97293f96e1d21175`
- `e05_embodied_vggt_step_260000.pt`：`ce5698a438b3f246d1d1fe0bcc4d1b205876c0523066cf11f08f8434d98770c5`

下载的数据可以使用现有安装脚本恢复训练目录。安装过程会校验分片摘要，并复制缓存、解压 RGB/action/meta；下载和安装副本并存时会需要约两份数据的磁盘空间，另加解压开销。

```bash
hf download "$HF_NAMESPACE/e05-robotwin-data" \
  --repo-type dataset --local-dir .cache/hf-verify/robotwin-data
python scripts/install_data.py .cache/hf-verify/robotwin-data --output .cache/hf-verify/installed-data
python scripts/validate_data.py --data-root .cache/hf-verify/installed-data --full \
  --report .cache/hf-verify/data_validation.json
```

推理使用下载的 109k 权重及其同目录统计；真机初始化将 `E05_EMBODIED_VGGT_CHECKPOINT` 指向下载的 VGGT-only 文件，并使用 `+model_init=embodied_vggt`。真实任务配置仍由对应任务提供。

## 8. 回填代码仓库的资源链接

发布成功后，再更新以下位置：

1. `README.md` 与 `README_zh-CN.md`：在 checkpoint 和数据章节放实际仓库链接，替换“尚未配置公开下载入口”的说明。
2. `configs/resources/checkpoint_manifest.json`：为两个资源填写真实 `repo_id`、下载 URL 和上传后确认的 revision；现有 SHA256 保持不变。
3. `docs/CHECKPOINTS.md`、`docs/DATA_CARD.md`、`docs/INSTALL.md`：写明获取方式与实际仓库名。
4. 两个 Model 仓库和 Dataset 仓库的 `README.md`：相互链接到代码、权重和数据资源。

可使用 `https://huggingface.co/账号/模型仓库/resolve/COMMIT_SHA/文件名` 形式记录固定版本权重地址；不要在实际 commit 尚未生成时编造 revision。`scripts/download_components.py` 仍只负责上游公开组件，新增 e05 资源可按本指南的 `hf download` 命令获取。
