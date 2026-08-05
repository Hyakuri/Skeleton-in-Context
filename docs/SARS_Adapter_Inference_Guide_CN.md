# Skeleton-in-Context 的 SARS-Inter 补全适配说明

## 物理边界

外部 SiC 进程只能读取 `completion_query.pkl`。`completion_evaluation_sidecar.pkl` 由 SARS-Inter 私有保存。query 只包含 H36M17/F64 残缺坐标、显式布尔 `missing_mask`、sample order 和坐标 metadata，不包含 HAR 标签、识别分数或 clean skeleton。

外部交换 PKL 固定使用 Pickle protocol 4，确保较新的 SARS-Inter Python 环境写出的文件可以由 Python 3.7 SiC 环境读取。

## F64 到 F16 策略

官方 SiC 模型的目标长度为 16 帧。适配器固定使用四个不重叠窗口：

```text
[0:16], [16:32], [32:48], [48:64]
```

每个窗口从官方 `3DPW_MC/train` 中确定性选择 demonstration，并把 query 的显式窗口 mask 同步应用到 demonstration input。模型预测只替换 `missing_mask=True` 的坐标，全部可见坐标都会被精确恢复。

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
- `coordinate_transform_mode`：当前只支持 `identity_h36m17`。

在 SiC 仓库根目录执行：

```powershell
<SIC_PYTHON> sars_adapter/run_completion.py
```

## 当前坐标限制

适配器已经显式记录坐标转换模式，但当前只实现 identity H36M17 传递。正式报告跨数据集结果前，必须核对各 SARS-Inter 数据集与官方 3DPW-MC 的轴方向、root 约定和尺度。新增 normalization 必须可逆，并完整写入结果 metadata。
