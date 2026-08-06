# Fair SiC Completion Pipeline for SARS-Inter

## Purpose

This guide runs Skeleton-in-Context as an isolated completion baseline for a SARS-Inter masked dataset. The external method receives only masked H36M17/F64 coordinates and a Boolean missing mask. Labels, clean skeletons, recognition scores, and evaluation metadata remain private to SARS-Inter.

The custom dataset and NW-UCLA must use the same frozen SiC checkpoint and method configuration. Their HAR checkpoints remain dataset-specific.

## Pipeline

```text
masked MMAction2 dataset
  -> SARS-Inter V2 query export
  -> isolated SiC completion
  -> SARS-Inter result import
  -> first/second HAR comparison
  -> visualization and paired statistics
```

## Required Frozen Inputs

- SiC repository commit and clean-worktree hash.
- SiC checkpoint SHA256.
- Official SiC source YAML SHA256.
- Official `3DPW_MC/train` prompt-pool manifest hash/count.
- SARS-Inter masked dataset and frozen recognizer checkpoint.
- Explicit source split, mask, model, and random seed.

Do not select prompts by sample ID, action label, dataset name, or evaluation result. The adapter derives prompt selection from `masked_keypoint + missing_mask` content and uses only the official train prompt pool.

## 1. Export The V2 Query In SARS-Inter

Configure `src/PP_ExportExternalCompletionInput.py` or call `export_external_completion_input()` with:

```python
{
    "dataset_path": "<MASKED_DATASET_PKL>",
    "dataset_profile": "custom",  # or "nw_ucla"
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

Pass only `completion_query.pkl` to the SiC process. Keep the sidecar in a separate SARS-Inter-only directory.

## 2. Run SiC Completion

Edit `build_direct_run_config()` in `sars_adapter/run_completion.py`:

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

Run:

```powershell
<SIC_PYTHON> sars_adapter/run_completion.py
```

Run the custom dataset and NW-UCLA separately. Change only query/output paths. Keep checkpoint, source config, data root, seed, mask policy, prompt policy, and coordinate mode identical.

## 3. Import The Result In SARS-Inter

Configure `src/PP_ImportExternalCompletionResults.py` with the source masked dataset, public query, private sidecar, SiC result, `source_split=test`, and `generated_split_name=generated_test`.

The importer validates query/result hashes and sample order, replaces only `missing_mask=True` coordinates, restores every observed coordinate exactly, and writes a new MMAction2 dataset. It does not overwrite the source dataset.

## 4. Recognition And Visualization

1. Run first recognition on the original masked `test` split.
2. Run second recognition on the imported dataset using the same dataset-specific recognizer checkpoint.
3. Use `selected_recognition_mode=derive_from_full` when `generated_test` is a subset of `test`.
4. Use `visualize_integration_strategy_dataset(..., strategy="external_completion")` to create raw/generated/final comparison videos.
5. Compare records by sample ID and report rescue, harm, stable-correct, stable-wrong, GT-probability change, and missing-joint reconstruction metrics.

## Smoke And Formal Runs

- A one-sample smoke run verifies interfaces only. It cannot estimate accuracy or significance, and t-SNE must be disabled.
- A formal run uses the complete frozen `test` split. Do not tune SiC settings or choose checkpoints on test results.
- Report out-of-distribution masks explicitly. `allow_ood_explicit` permits execution but does not claim official SiC mask support.
- Do not add method-specific smoothing or scale correction merely to improve figures. Any post-processing must be declared and applied as a separate ablation.

## Acceptance Checklist

- Query contains no evaluation fields.
- Result passes SARS-Inter V2 binding validation.
- Output is finite and has shape `(N, 64, 17, 3)`.
- Observed-restoration maximum absolute error is exactly zero.
- Every F64 sample uses one train prompt across four F16 windows.
- Repository and checkpoint identities are recorded.
- Before/after recognition uses the same recognizer checkpoint.
- Visualization manifest has no warnings.

## Batch Series Runner

Formal multi-dataset and multi-mask execution uses
`sars_adapter/run_completion_series.py`. SARS-Inter creates one public
`external_completion_run_plan.json`; the SiC process reads only that plan and
the referenced public queries. It never receives the private evaluation
sidecars.

Edit `build_direct_run_config()` in the series runner:

| Parameter | Meaning |
| --- | --- |
| `plan_path` | Public run plan generated by SARS-Inter. It contains query paths, query hashes, public identities, and relative result paths only. |
| `output_root` | Root under which every job's `result_relative_path` is materialized. |
| `checkpoint_path` | Frozen SiC checkpoint. Its SHA256 is recorded and rechecked before the final summary is written. |
| `source_config` | Official SiC YAML used by the checkpoint. Its content hash is included in prompt-pool provenance. |
| `data_root` | SiC data root containing the train-only demonstration pool. |
| `device` | Inference device, normally `cuda:0`. |
| `dry_run` | `True` validates the plan without loading the model; `False` performs inference. |
| `require_clean_repository` | Formal default `True`; rejects an adapter worktree with uncommitted changes. |
| `resume` | Reuses an existing result only after complete query, shape, finite-value, checkpoint, repository, and provenance validation. |
| `strict` | Raises after a failed job. A failure summary is still saved first. |
| `continue_on_error` | Continues later jobs only when `strict=False`; never turns a failed record into success. |
| `mask_policy` | `strict_official_mc` rejects unsupported masks; `allow_ood_explicit` runs the paper masks and records explicit OOD reasons. |
| `demonstration_seed` | Deterministic train-demonstration selection seed. |
| `demonstration_selection_policy` | `per_sample_fixed` uses one train demonstration across the four F16 windows of a sample. |
| `coordinate_transform_mode` | Formal mode is `project_h36m17_prompt_aligned_v1`; it applies the frozen joint, axis, root, and scale adapter. |

Run:

```powershell
<SIC_PYTHON> sars_adapter/run_completion_series.py
```

Progress is written atomically to a status JSON and printed as
`[current/total] job_id=... status=...`. The model is loaded once per series.
If checkpoint, prompt-pool, source-config, repository, query, or existing-result
content changes during the run, the series fails instead of publishing mixed
provenance.

## Completion Scopes

- `all_split` is the primary completion-method comparison. One SiC result is
  produced per dataset, mask, and split, then SARS-Inter fans that result out to
  all configured dataset-specific HAR models. First recognition supplies the
  Before metric but does not gate SiC.
- `trigger_selected` is a separate system-level supplement. Its query is bound
  to a frozen validation-margin or validation-threshold selection manifest and
  is produced per dataset, mask, model, and split.

Never combine both scopes in one result identity or report a
`trigger_selected` result as an all-sample completion baseline.

