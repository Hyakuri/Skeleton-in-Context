# Skeleton-in-Context 的 SARS-Inter 补全适配说明

## 物理边界

外部 SiC 进程只能读取 `completion_query.pkl`。`completion_evaluation_sidecar.pkl` 由 SARS-Inter 私有保存。query 只包含 H36M17/F64 残缺坐标、显式布尔 `missing_mask`、sample order 和坐标 metadata，不包含 HAR 标签、识别分数或 clean skeleton。

外部交换 PKL 固定使用 Pickle protocol 4，确保较新的 SARS-Inter Python 环境写出的文件可以由 Python 3.7 SiC 环境读取。

## F64 到 F16 策略

官方 SiC 模型的目标长度为 16 帧。适配器固定使用四个不重叠窗口：

```text
[0:16], [16:32], [32:48], [48:64]
```

正式比较模式为每个 F64 sample 从官方 `3DPW_MC/train` 中确定性选择一个 demonstration，四个窗口重复使用同一个 demonstration。Query 的显式窗口 mask 会同步应用到 demonstration input。模型预测只替换 `missing_mask=True` 的坐标，全部可见坐标都会被精确恢复。

适配器会报告第 16、32、48 帧边界的位置和速度不连续度，但不会对生成关节做平滑或插值，避免给 SiC 增加其他补值方法没有使用的额外后处理。

## Mask 模式

- `strict_official_mc`：只接受每个 16 帧窗口内保持不变、恰好遮挡 6 或 10 个关节（官方 `drop_ratios_MC=[0.4,0.6]` 支持范围）、且不遮挡关节 0 和 16 的 mask。
- `allow_ood_explicit`：允许执行任意显式 F64 mask，但会记录其偏离官方 MC 训练分布的原因。

除非追加匹配的 SiC 训练协议，否则 temporal interruption 结果必须表述为 SiC 的分布外应用。

## Checkpoint 来源模式

两种模式共用完全相同的后续补全逻辑：

- `direct_path` 用于冒烟和诊断。填写 `checkpoint_path` 与匹配的 `source_config` 后，程序会自动计算 checkpoint SHA256；dry-run 会报告该哈希，但不会加载模型。
- `identity_manifest` 是正式实验默认模式。它只接受 `checkpoint_identity_policy=require_mc_only` 的可移植 `checkpoint_identity.json`，并绑定 MC-only checkpoint、对应的 `effective_config.yaml`、文件大小、SHA256、训练身份和 manifest 自身哈希。

正式冻结时，在 `sars_adapter/freeze_checkpoint.py` 的 `build_direct_run_config()` 中填写参数。先使用 `dry_run=True` 检查身份，再改为 `dry_run=False` 发布 bundle：

```text
<bundle>/
  checkpoint_identity.json
  <selected-checkpoint>.bin
  effective_config.yaml
```

manifest 只保存相对文件名。跨电脑时整体复制该目录，只需填写新电脑上的 `checkpoint_identity_manifest_path`，不需要复制旧绝对路径，也不需要手工重新输入 SHA256。已有不同 bundle 即使旧配置仍写有 `overwrite=True` 也不会被原地替换；应使用新的输出目录。

## 直接执行参数

在 `sars_adapter/run_completion.py` 的 `build_direct_run_config()` 中调整：

- `dry_run`：只检查路径/query，或执行真实推理。
- `query_path`：V2 `completion_query.pkl`，不得指向私有 sidecar。
- `checkpoint_source_mode`：正式运行填写 `identity_manifest`，冒烟运行填写 `direct_path`。
- `checkpoint_identity_manifest_path`：仅正式模式填写，指向复制后 bundle 内的 `checkpoint_identity.json`。
- `checkpoint_path`：仅 `direct_path` 填写，SHA256 由程序自动计算。
- `source_config`：仅 `direct_path` 填写，必须是 checkpoint 同一次训练保存的 `effective_config.yaml`。
- `checkpoint_identity_policy`：正式 manifest 模式固定为 `require_mc_only`；`allow_legacy` 只允许配合 `direct_path` 复现旧冒烟权重，不得用于论文正式结果。
- `data_root`：包含 `3DPW_MC/train` 的官方数据根目录。
- `output_path`：返回 SARS-Inter 的 V2 `completion_result.pkl`。
- `device`：`cuda:0` 或 `cpu`。
- `mask_policy`：官方 MC 严格支持或显式 OOD 研究模式。
- `demonstration_seed`：train-only demonstration 的确定性选择 seed。
- `demonstration_selection_policy`：正式比较使用 `per_sample_fixed`，四个窗口共享一个 train demonstration。
- `coordinate_transform_mode`：正式 SARS-Inter 比较使用 `project_h36m17_prompt_aligned_v1`。该字符串只保留为跨仓库数据契约标识；适配器内部只有当前实现，不再提供旧 root-only 实现的选择分支。`identity_h36m17` 仅用于复现历史结果。

这两个参数被有意绑定：正式坐标模式必须使用 `per_sample_fixed`，旧版 `identity_h36m17` 必须使用 `per_window`。组合不匹配时程序会明确报错，不会静默改变历史行为。

在 SiC 仓库根目录执行：

```powershell
<SIC_PYTHON> sars_adapter/run_completion.py
```

## 正式坐标适配

`project_h36m17_prompt_aligned_v1` 对自获数据和 NW-UCLA 使用完全相同的确定性策略：

Query 的坐标契约必须显式声明 `joint_order=h36m17_sars_inter_project_order` 与 `axis_order=[x_lateral,y_depth,z_height]`。这样可以防止已经采用标准 H36M17 顺序的数据再次发生左右肢体置换。

1. 把项目的左腿/右腿和右臂/左臂分组置换到 SiC/MotionBERT H36M17 顺序。
2. 使用 `(x, y_depth, z_height) -> (x, z_height, -y_depth)` 把项目 Z-up 坐标旋转到 SiC Y-up 空间。
3. 只根据 query 可见坐标，为整个 F64 sample 确定一种语义锚点。固定优先级为 pelvis/双髋中点、中心躯干、上躯干、双肩中点、固定排序可见关节质心；train demonstration 使用语义相同的关节或关节集合。
4. 对缺失的锚点帧进行时间插值，并在序列边界使用最近的可见锚点延伸；四个 F16 窗口共享同一条 F64 锚点轨迹和同一个可见骨段尺度。
5. 将 query 对齐到选定的 train demonstration，运行四个 F16 窗口，再对输出执行逆变换。
6. 精确恢复项目空间中的全部可见坐标。

结果会记录关节置换、轴矩阵、语义锚点模式/关节/hash、sample/reference scale、prompt identity/hash、prompt 池 manifest hash/count、source config hash、checkpoint 训练身份、缺失关节窗口边界诊断、仓库身份和 checkpoint SHA256。稳定的 completion policy 会写入 `coordinate_anchor_policy=paired_visible_semantic_anchor`，因此 `resume=True` 不会静默复用旧 root-only 适配器生成的结果。Prompt 选择只使用 `masked_keypoint + missing_mask` 的稳定哈希，不解析 sample ID、标签或数据集名称。

同时遮挡 pelvis 和双髋的 `bottom` 不属于 SiC 官方 MC 随机 mask 分布。使用 `mask_policy=allow_ood_explicit` 时，适配器会使用语义对应的可见躯干或肩部锚点，并继续记录 OOD 原因；这只表示允许完成公平比较，不代表把该遮挡描述为分布内样本。若整个 sample 没有可用语义锚点，或可见骨段不足以估计尺度，程序仍会明确拒绝。

## 自获数据与 NW-UCLA 运行方式

每个 query 分别运行一次 `sars_adapter/run_completion.py`。两个数据集必须使用相同的冻结 checkpoint、`source_config`、`data_root`、`demonstration_seed`、`mask_policy`、demonstration selection policy 和 coordinate mode；只修改 `query_path` 与 `output_path`。

每份结果保存到新的方法专用 run 目录。SiC 只能接收 query 文件路径，不得让它访问包含 `completion_evaluation_sidecar.pkl` 的父目录。结果返回 SARS-Inter 后，分别使用两个数据集各自冻结的识别 checkpoint 导入和评价；两个数据集的结果分开报告，不合并准确率。
