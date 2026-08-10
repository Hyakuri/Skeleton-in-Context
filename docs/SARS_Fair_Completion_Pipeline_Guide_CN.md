# SARS-Inter 公平 SiC 骨骼补全流程

## 目标与边界

本流程将 Skeleton-in-Context（SiC）作为独立的骨骼补全比较方法。SiC 只接收
公开的 H36M17/F64 残缺骨骼和 `missing_mask`，不会接收 label、clean skeleton、
第一次识别分数、二次识别结果或私有评价 sidecar。

自获数据集和 NW-UCLA 使用同一个冻结 SiC checkpoint、同一套 train-only
demonstration、同一坐标适配和同一推理参数；HAR checkpoint 仍按数据集分别使用。

公开 run plan 必须包含 `method_profile.checkpoint_sha256`，并与
`checkpoint_path` 文件的 64 位 SHA256 完全一致。缺失或不一致时，series runner
会在加载模型前停止。

```text
SARS-Inter masked dataset
  -> 公开 completion_query.pkl 与公开 run plan
  -> 隔离的 SiC series runner
  -> completion_result.pkl
  -> SARS-Inter 私有导入、二次识别、统计和可视化
```

## 必须冻结的身份

- SiC adapter fork 和官方 upstream 的 URL、commit、dirty/worktree 状态仅作为追踪信息。
- 可移植 SiC checkpoint identity bundle 及其 manifest hash。
- 由 manifest 记录的 SiC checkpoint 与本次训练 `effective_config.yaml` SHA256。
- `3DPW_MC/train` prompt pool 的文件数量和 manifest SHA256。
- demonstration seed、选择策略、mask policy 和坐标转换模式。

正式运行会严格复核 checkpoint、配置、prompt pool 和数据契约，避免在同一批结果中
混入不同实验资产。仓库 revision 与 dirty 状态只记录并提示，不阻断实验执行。

## 第一步：SARS-Inter 导出

推荐通过 `src/PP_Informatics_MajorRevisionPipeline.py` 执行。pipeline 会生成：

```text
external_completion_export/
  public_query/.../completion_query.pkl
  public_query/external_completion_run_plan.json
  private_evaluation/.../completion_evaluation_sidecar.pkl
```

只把 `public_query` 目录交给 SiC。`private_evaluation` 必须留在 SARS-Inter 侧。

`completion_query.pkl` 固定包含：

- `masked_keypoint`: `(N,64,17,3)`、`float32`。
- `missing_mask`: `(N,64,17)`、`bool`，`True=missing`。
- H36M17 项目关节顺序、Z-up 坐标契约和 sample order。
- dataset、mask、split、scope 与 query hash 等公开身份。

## 第二步：SiC 批量补全

正式多组合运行使用 `sars_adapter/run_completion_series.py`，而不是逐个修改
`run_completion.py`。

在 `build_direct_run_config()` 中设置：

| 参数 | 可填内容与说明 |
| --- | --- |
| `plan_path` | SARS-Inter 生成的 `external_completion_run_plan.json`。 |
| `output_root` | 结果根目录；每个 job 按 plan 的 `result_relative_path` 保存。 |
| `checkpoint_source_mode` | 正式跨设备运行填写 `identity_manifest`；`direct_path` 只用于冒烟诊断。 |
| `checkpoint_identity_manifest_path` | 正式 bundle 内的 `checkpoint_identity.json`；程序会解析相对 checkpoint/config 文件名并验证全部哈希。 |
| `checkpoint_path` | 仅 `direct_path` 使用，SHA256 由程序自动计算。 |
| `source_config` | 仅 `direct_path` 使用，必须与所选 checkpoint 来自同一次训练。 |
| `checkpoint_identity_policy` | 正式 manifest 模式固定为 `require_mc_only`；`allow_legacy` 只允许 direct-path 旧冒烟 checkpoint。 |
| `data_root` | SiC 数据根目录，必须包含 train-only demonstration pool。 |
| `device` | 通常为 `cuda:0`；CPU 仅适合接口测试。 |
| `dry_run` | `True` 校验 plan 与 checkpoint 身份且不加载模型；`False` 运行真实 GPU 推理。 |
| `resume` | `True` 时仅复用通过 query、shape、finite、checkpoint、prompt pool、source config 和 completion policy 校验的已有结果；completion policy 包含语义锚点身份，因此旧 root-only 适配器结果会被拒绝；Git revision 差异不使结果失效。 |
| `strict` | `True` 时 job 失败后先保存 summary，再抛出异常。 |
| `continue_on_error` | 仅在 `strict=False` 时继续后续 job。 |
| `mask_policy` | `strict_official_mc` 拒绝官方 MC 分布外 mask；`allow_ood_explicit` 允许论文遮挡并记录 OOD 原因。 |
| `demonstration_seed` | train demonstration 的确定性种子。 |
| `demonstration_selection_policy` | 正式使用 `per_sample_fixed`，同一 F64 sample 的四个 F16 window 共用一个 demonstration。 |
| `coordinate_transform_mode` | 正式值继续使用 `project_h36m17_prompt_aligned_v1` 作为跨仓库契约标识。其唯一当前实现执行冻结的关节/轴转换、成对可见语义锚点和一个 F64 可见骨段尺度，不再提供旧 root-only 实现的选择分支。 |

对于每个 F64 sample，适配器按固定顺序选择 query/prompt 成对锚点：pelvis 或双髋中点、中心躯干、上躯干、双肩中点、固定排序可见关节质心。这样可以在不伪造 pelvis 观测、也不读取 clean/GT 数据的前提下运行显式 OOD 的 `bottom` mask。四个 F16 窗口共享同一条锚点轨迹和尺度；完全不可观测或几何支持不足的样本仍会明确失败。

执行：

```powershell
<SIC_PYTHON> sars_adapter/run_completion_series.py
```

终端会输出：

```text
[当前/总数] job_id=... status=running|completed|skipped_existing|failed
```

模型在同一个 series 中只加载一次。status 与 summary JSON 使用原子写入，可用于
中断恢复和失败审计。

## 第三步：SARS-Inter 恢复运行

把 SiC 的 `output_root` 配置为 SARS-Inter external import 所记录的结果根目录，
保持原 pipeline `output_run_name` 不变并再次执行。已完成的 export 会恢复，import
会校验结果，然后继续第二次识别、统计和可视化。

导入器会：

1. 校验 query hash、sample order、dataset/split、H36M17/F64 和有限值。
2. 严格校验 checkpoint、prompt pool、source config 和 completion policy；adapter/upstream revision 仅记录。
3. 只替换 `missing_mask=True` 的坐标。
4. 精确恢复所有原始可见坐标，误差必须为 0。
5. 保存 method、scope、runtime、mask policy、OOD 原因和完整 provenance。

## 两种 Scope

### `all_split`

这是论文中 SiC 补全方法比较的主实验。每个 dataset/mask/split 只运行一次 SiC，
然后 SARS-Inter 将同一补全结果分发给所有指定的、数据集专用的 HAR 模型。第一
次识别只提供 Before 指标，不作为是否补全的条件。

### `trigger_selected`

这是独立的系统级补充实验。query 必须绑定冻结的 validation margin 或 validation
threshold manifest，并按 dataset/mask/model/split 分别处理。不得把该结果表述为
全样本补全基线。

## OOD 与公平比较

SiC 官方 MC mask 与本项目长时间遮挡并不完全一致。正式实验可使用
`allow_ood_explicit`，但必须报告 `official_mc_compatible` 和
`mask_ood_reasons`。这表示“允许执行并审计”，不表示官方训练分布覆盖该 mask。

不得根据 test label、补全精度、HAR after 结果或可视化效果选择 demonstration、
checkpoint 或推理参数。不得仅为 SiC 增加未对其他方法使用的 smoothing、scale
correction 或后处理。

## 验收清单

- public query 不含 label、clean skeleton、sidecar 路径或识别分数。
- 每个结果通过 query hash 和 sample order 绑定。
- 输出 shape 为 `(N,64,17,3)` 且全部为有限值。
- 可见坐标恢复最大绝对误差为 0。
- demonstration 只来自 train。
- adapter、upstream、checkpoint、配置和 prompt pool 身份完整。
- all_split 与 trigger_selected 分开保存、统计和表述。
- Before/After 使用同一个数据集专用 HAR checkpoint。
- 外部补全可视化使用导入后的 raw/generated/final 三联结果。
