# 面向 SARS-Inter 评价的 SiC 子集训练说明

## 目的

本适配器只使用选定的 SiC 官方骨骼任务训练 Skeleton-in-Context，不读取 HAR 类别、识别分数、Utility Bank 或目标数据集 clean skeleton。因此，训练完成的 checkpoint 与 SARS-Inter 识别模型无关，可以由所有下游 HAR 模型复用。

针对 SARS-Inter 骨骼补值比较，推荐选择代码中的 `MC`。它在论文中对应 **Joint Completion（JC）**。论文中应将全量 MC 训练结果表述为“本地 MC-only 训练的 SiC 基线”；如果设置了样本上限，则表述为“本地 MC-only 子集训练的 SiC 基线”。两者都不能写成“官方预训练 checkpoint”。

## 为什么选择 MC

| 代码任务 | 官方任务含义 | 与本项目的关系 |
| --- | --- | --- |
| `PE` | 2D 到 3D 姿态估计 | 不学习缺失关节补全。 |
| `MP` | 根据历史动作预测未来动作 | 处理未来帧，不是空间关节缺失。 |
| `MC` | 在 16 帧内持续缺失 40% 或 60% 关节的 Joint Completion | 与本项目长时间关节遮挡最接近。 |
| `FPE` | 根据未来 2D 观测估计未来 3D 姿态 | 不等价于显式 3D 缺失关节补全。 |

官方 MC loader 使用 0.4 和 0.6 两种比例，实际遮挡 6 个或 10 个关节。候选范围严格为关节索引 1 到 15，索引 0 和 16 均被排除；同一组关节会在全部 16 帧中持续置零。这是最接近本项目的官方监督，但并不覆盖所有 SARS-Inter mask。关节集合随时间变化、只覆盖部分窗口、或缺失关节数量不在官方范围内的情况仍属于 OOD，必须继续保留 adapter 的诊断记录。

## 物理隔离边界

以下资源应保持分离：

- SiC 上游源码仓库；
- 只安装上游依赖的 Python 3.7 独立环境；
- 外部数据与 checkpoint 根目录；
- SARS-Inter query/result 交换产物。

不要把 SiC 依赖复制到 SARS-Inter 或 OmniControl 环境。

## 官方数据

MC-only 训练时，数据根目录只需要包含：

```text
<SIC_DATA_ROOT>/
  3DPW_MC/train, 3DPW_MC/test
  support_data/
```

启用其他任务时，才需要补充相应的官方任务目录。训练 query 和 demonstration 只从官方 train 目录选择，子集选择器不会裁剪 test 目录。

## 直接执行参数

在 `sars_adapter/train_subset.py` 的 `build_direct_run_config()` 中调整：

- `dry_run`：`True` 只校验并打印计划；`False` 开始真实 GPU 训练。
- `data_root`：官方 ready-to-use 数据根目录。
- `checkpoint_root`：checkpoint 与 manifest 输出根目录。
- `run_name`：本次 run 名，支持 `{timestamp}`。
- `tasks`：`PE, MP, MC, FPE` 的非空有序子集；本次比较使用 `['MC']`。
- `train_sample_limits`：必须只填写已启用任务。`None` 表示使用该任务全部 train 文件，正整数表示确定性抽取对应数量。
- `subset_seed`：控制文件名子集选择。
- `epochs`：训练 epoch 数。
- `batch_size`：单 GPU 物理 batch size。
- `test_batch_size`：后续官方评价 batch size。
- `num_workers`：DataLoader worker 数；Windows 首次建议从 `0` 开始。正式训练可尝试 `2` 或 `4`，但必须同时设置 `persistent_workers=False`。
- `persistent_workers`：严格可恢复训练必须保持 `False`。这样每个 epoch 都会根据已恢复的主 Torch RNG 重新创建 worker；常驻 worker 自身的 RNG 状态无法写入 checkpoint。
- `no_eval`：子集训练时建议保持 `True`，训练结束后单独评价最终 checkpoint。
- `seed`：模型和训练主随机种子。

每次真实训练都会在 checkpoint 旁保存 `effective_config.yaml` 和 `training_subset_manifest.json`。manifest 会记录 `task_scope=single_task/multi_task` 和实际选择的全部文件。
其中 `effective_config.yaml` 会显式保存已经解析的 `data` 配置，保证 checkpoint 可以脱离训练过程中的临时变量重新加载。

## 推荐执行顺序

1. 设置 `tasks=['MC']`、`train_sample_limits={'MC': 64}`、1 epoch、B8，执行 dry run。
2. 使用相同配置运行一轮 GPU smoke test。
3. 检查耗时、峰值显存、loss 有限性和 checkpoint round trip。
4. 固定 MC pilot 子集规模，例如 4000 条；若要使用全部官方 MC train 文件，设置为 `{'MC': None}`。
5. 使用固定 seed 训练，并记录最终 checkpoint SHA256。
6. 只在官方 MC test 数据上评价 checkpoint。
7. 自获数据集与 NW-UCLA 共用同一冻结 checkpoint。

在 SiC 仓库根目录执行：

```powershell
<SIC_PYTHON> sars_adapter/train_subset.py
```

训练完成后执行上游评价：

```powershell
<SIC_PYTHON> train.py --config <RUN>/effective_config.yaml --checkpoint <RUN> --evaluate <RUN>/latest_epoch.bin
```

只有在一轮 smoke test 获得本机实际耗时和安全 batch size 后，才能启动 120 epochs 正式训练。

`no_eval=True` 时，训练保存 `latest_epoch.bin`，不会在训练期间反复遍历完整 MC test；训练结束后再独立评价一次。`no_eval=False` 时会额外保存 `best_epoch_MC.bin`，但训练期间会多次执行官方 MC 评价，耗时明显增加。单任务训练还会生成 `best_epoch_all.bin`，它与 MC 最优选择数值相同，仅用于保持历史文件命名兼容。

正式评价和后续补值必须使用 run 目录中的 `effective_config.yaml`，不要改回仓库默认 YAML。这样 checkpoint provenance 才能明确绑定 MC-only 任务范围和实际训练参数。正式补值还必须设置 `checkpoint_identity_policy=require_mc_only`，适配器会比较 checkpoint 中记录的 effective-config SHA256，并拒绝旧权重或多任务权重。若训练中断，可执行 `train.py --config <RUN>/effective_config.yaml --checkpoint <RUN>` 从同一目录恢复；上游训练循环会自动发现 `<RUN>/latest_epoch.bin`。

一轮 epoch 的 smoke checkpoint 只用于证明运行环境和训练闭环有效，不能用于报告正式补全精度。
