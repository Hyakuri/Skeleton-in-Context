# SARS-Inter 公平 SiC 骨骼补值流程指南

## 目的

本指南用于把 Skeleton-in-Context 作为与 SARS-Inter 物理隔离的骨骼补值比较方法。外部方法只能读取 H36M17/F64 残缺骨骼和布尔 `missing_mask`。标签、clean skeleton、识别分数和评价 metadata 始终由 SARS-Inter 私有保存。

自获数据与 NW-UCLA 必须使用同一个冻结的 SiC checkpoint 和完全相同的方法配置。两者的动作识别 checkpoint 仍按各自数据集分别使用。

## 整体流程

```text
masked MMAction2 dataset
  -> SARS-Inter V2 query 导出
  -> 隔离的 SiC 补值
  -> SARS-Inter 结果导入
  -> 第一次/第二次识别比较
  -> 可视化和配对统计
```

## 必须冻结的输入

- SiC repository commit 与 clean-worktree hash。
- SiC checkpoint SHA256。
- 官方 SiC source YAML SHA256。
- 官方 `3DPW_MC/train` prompt 池 manifest hash/count。
- SARS-Inter masked dataset 与冻结的识别 checkpoint。
- 明确的 source split、mask、model 和随机种子。

不得根据 sample ID、动作标签、数据集名称或评价结果选择 prompt。适配器只根据 `masked_keypoint + missing_mask` 内容确定性选择 prompt，并且只读取官方 train prompt 池。

## 1. 在 SARS-Inter 导出 V2 Query

在 `src/PP_ExportExternalCompletionInput.py` 中配置，或调用 `export_external_completion_input()`：

```python
{
    "dataset_path": "<MASKED_DATASET_PKL>",
    "dataset_profile": "custom",  # 或 "nw_ucla"
    "source_split": "test",
    "completion_scope": "all_split",
    "schema_mode": "v2_physical_split",
    "coordinate_contract": {
        "skeleton": "H36M17",
        "joint_order": "h36m17_sars_inter_project_order",
        "axis_order": ["x_lateral", "y_depth", "z_height"],
        "unit": "dataset_normalized",
    },
    "query_output_path": "<PUBLIC_QUERY_DIR>/completion_query.pkl",
    "evaluation_sidecar_output_path": (
        "<PRIVATE_EVALUATION_DIR>/completion_evaluation_sidecar.pkl"
    ),
}
```

SiC 进程只能接收 `completion_query.pkl`。Sidecar 必须保存在另一个仅由 SARS-Inter 读取的目录中。

## 2. 运行 SiC 补值

在 `sars_adapter/run_completion.py` 的 `build_direct_run_config()` 中设置：

```python
{
    "dry_run": False,
    "query_path": "<PUBLIC_QUERY_DIR>/completion_query.pkl",
    "checkpoint_path": "<FROZEN_SIC_CHECKPOINT>",
    "source_config": "<SIC_REPOSITORY>/configs/default.yaml",
    "data_root": "<SIC_DATA_ROOT>",
    "output_path": "<METHOD_RESULT_DIR>/completion_result.pkl",
    "device": "cuda:0",
    "mask_policy": "allow_ood_explicit",
    "demonstration_seed": 42,
    "demonstration_selection_policy": "per_sample_fixed",
    "coordinate_transform_mode": "project_h36m17_prompt_aligned_v1",
}
```

执行：

```powershell
<SIC_PYTHON> sars_adapter/run_completion.py
```

自获数据和 NW-UCLA 各运行一次。只修改 query 与 output 路径；checkpoint、source config、data root、seed、mask policy、prompt policy 和 coordinate mode 必须保持一致。

## 3. 在 SARS-Inter 导入结果

在 `src/PP_ImportExternalCompletionResults.py` 中填写源 masked dataset、公开 query、私有 sidecar、SiC result，并设置 `source_split=test`、`generated_split_name=generated_test`。

导入器会校验 query/result hash 和 sample order，只替换 `missing_mask=True` 的坐标，精确恢复全部可见坐标，并写出新的 MMAction2 dataset。源数据集不会被覆盖。

## 4. 识别与可视化

1. 对原始 masked dataset 的 `test` split 执行第一次识别。
2. 对导入数据集执行第二次识别，并使用同一数据集对应的同一个识别 checkpoint。
3. 当 `generated_test` 是 `test` 的子集时，使用 `selected_recognition_mode=derive_from_full`。
4. 使用 `visualize_integration_strategy_dataset(..., strategy="external_completion")` 生成 raw/generated/final 三联视频。
5. 按 sample ID 比较结果，报告 rescue、harm、stable-correct、stable-wrong、GT 概率变化和缺失关节重建指标。

## Smoke 与正式实验

- 单样本 smoke 只能验证接口，不能估计精度或显著性；t-SNE 必须关闭。
- 正式实验必须使用冻结的完整 `test` split，不得根据 test 结果调整 SiC 参数或选择 checkpoint。
- 必须明确报告 OOD mask。`allow_ood_explicit` 只表示允许执行，不表示该 mask 属于 SiC 官方训练支持范围。
- 不得为了改善图像而静默增加 SiC 专用 smoothing 或 scale correction。任何后处理都必须作为独立消融公开报告。

## 验收清单

- Query 不含任何评价字段。
- Result 通过 SARS-Inter V2 binding 校验。
- 输出为有限值，shape 为 `(N, 64, 17, 3)`。
- 可见坐标恢复最大绝对误差严格为 0。
- 每个 F64 sample 的四个 F16 窗口共用一个 train prompt。
- 仓库与 checkpoint 身份已记录。
- Before/after 识别使用同一个识别 checkpoint。
- 可视化 manifest 不含 warning。

