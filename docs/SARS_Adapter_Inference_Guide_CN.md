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

## 直接执行参数

在 `sars_adapter/run_completion.py` 的 `build_direct_run_config()` 中调整：

- `dry_run`：只检查路径/query，或执行真实推理。
- `query_path`：V2 `completion_query.pkl`，不得指向私有 sidecar。
- `checkpoint_path`：冻结的 SiC checkpoint。
- `source_config`：与训练一致的 SiC YAML。
- `data_root`：包含 `3DPW_MC/train` 的官方数据根目录。
- `output_path`：返回 SARS-Inter 的 V2 `completion_result.pkl`。
- `device`：`cuda:0` 或 `cpu`。
- `mask_policy`：官方 MC 严格支持或显式 OOD 研究模式。
- `demonstration_seed`：train-only demonstration 的确定性选择 seed。
- `demonstration_selection_policy`：正式比较使用 `per_sample_fixed`，四个窗口共享一个 train demonstration。
- `coordinate_transform_mode`：正式 SARS-Inter 比较使用 `project_h36m17_prompt_aligned_v1`；`identity_h36m17` 只用于复现历史结果。

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
3. 只根据 query 可见坐标估计一条 F64 root 轨迹和一个可见骨段尺度。
4. 将 query 对齐到选定的 train demonstration，运行四个 F16 窗口，再对输出执行逆变换。
5. 精确恢复项目空间中的全部可见坐标。

结果会记录关节置换、轴矩阵、root hash、sample/reference scale、prompt identity/hash、prompt 池 manifest hash/count、source config hash、缺失关节窗口边界诊断、仓库身份和 checkpoint SHA256。Prompt 选择只使用 `masked_keypoint + missing_mask` 的稳定哈希，不解析 sample ID、标签或数据集名称。

## 自获数据与 NW-UCLA 运行方式

每个 query 分别运行一次 `sars_adapter/run_completion.py`。两个数据集必须使用相同的冻结 checkpoint、`source_config`、`data_root`、`demonstration_seed`、`mask_policy`、demonstration selection policy 和 coordinate mode；只修改 `query_path` 与 `output_path`。

每份结果保存到新的方法专用 run 目录。SiC 只能接收 query 文件路径，不得让它访问包含 `completion_evaluation_sidecar.pkl` 的父目录。结果返回 SARS-Inter 后，分别使用两个数据集各自冻结的识别 checkpoint 导入和评价；两个数据集的结果分开报告，不合并准确率。
