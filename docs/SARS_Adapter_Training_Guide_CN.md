# 面向 SARS-Inter 评价的 SiC 子集训练说明

## 目的

本适配器只使用 SiC 官方 PE、MP、MC、FPE 骨骼数据训练 Skeleton-in-Context，不读取 HAR 类别、识别分数、Utility Bank 或目标数据集 clean skeleton。因此，训练完成的 checkpoint 与 SARS-Inter 识别模型无关，可以由所有下游 HAR 模型复用。

论文中必须将该结果表述为“本地子集训练的 SiC 基线”，不能表述为“官方预训练 checkpoint”。

## 物理隔离边界

以下资源应保持分离：

- SiC 上游源码仓库；
- 只安装上游依赖的 Python 3.7 独立环境；
- 外部数据与 checkpoint 根目录；
- SARS-Inter query/result 交换产物。

不要把 SiC 依赖复制到 SARS-Inter 或 OmniControl 环境。

## 官方数据

数据根目录必须包含：

```text
<SIC_DATA_ROOT>/
  H36M/train, H36M/test
  AMASS/train, AMASS/test
  3DPW_MC/train, 3DPW_MC/test
  H36M_FPE/train, H36M_FPE/test
  source_data/H36M.pkl
  support_data/
```

训练 query 和 demonstration 只从上述官方 train 目录选择。子集选择器不会裁剪 test 目录。

## 直接执行参数

在 `sars_adapter/train_subset.py` 的 `build_direct_run_config()` 中调整：

- `dry_run`：`True` 只校验并打印计划；`False` 开始真实 GPU 训练。
- `data_root`：官方 ready-to-use 数据根目录。
- `checkpoint_root`：checkpoint 与 manifest 输出根目录。
- `run_name`：本次 run 名，支持 `{timestamp}`。
- `tasks`：必须保持 `PE, MP, MC, FPE` 四任务。
- `train_sample_limits`：每个官方训练任务稳定选择的 query/prompt 数量。
- `subset_seed`：控制文件名子集选择。
- `epochs`：训练 epoch 数。
- `batch_size`：单 GPU 物理 batch size。
- `test_batch_size`：后续官方评价 batch size。
- `num_workers`：DataLoader worker 数；Windows 首次建议从 `0` 开始。
- `no_eval`：子集训练时建议保持 `True`，训练结束后单独评价最终 checkpoint。
- `seed`：模型和训练主随机种子。

每次真实训练都会在 checkpoint 旁保存 `effective_config.yaml` 和 `training_subset_manifest.json`。
其中 `effective_config.yaml` 会显式保存已经解析的 `data` 配置，保证 checkpoint 可以脱离训练过程中的临时变量重新加载。

## 推荐执行顺序

1. 每任务 64 条、1 epoch、B8 执行 dry run。
2. 使用相同配置运行一轮 GPU smoke test。
3. 检查耗时、峰值显存、loss 有限性和 checkpoint round trip。
4. 固定 pilot 子集规模，例如每任务 1000 条。
5. 使用固定 seed 训练，并记录最终 checkpoint SHA256。
6. 在官方 test 任务上评价 checkpoint。
7. 自获数据集与 NW-UCLA 共用同一冻结 checkpoint。

在 SiC 仓库根目录执行：

```powershell
<SIC_PYTHON> sars_adapter/train_subset.py
```

训练完成后执行上游评价：

```powershell
<SIC_PYTHON> train.py --config <RUN>/effective_config.yaml --evaluate <RUN>/latest_epoch.bin
```

只有在一轮 smoke test 获得本机实际耗时和安全 batch size 后，才能启动 120 epochs 正式训练。

一轮 epoch 的 smoke checkpoint 只用于证明运行环境和训练闭环有效，不能用于报告正式补全精度。
