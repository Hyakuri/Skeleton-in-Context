# SARS-Inter Completion Adapter for Skeleton-in-Context

## Boundary

The external process reads only `completion_query.pkl`. SARS-Inter keeps `completion_evaluation_sidecar.pkl` private. The query contains masked H36M17/F64 coordinates, an explicit Boolean `missing_mask`, sample order, and coordinate metadata. It contains no HAR labels, recognition scores, or clean skeleton.

External exchange PKL files use Pickle protocol 4 so the Python 3.7 SiC environment can read artifacts written by a newer SARS-Inter environment.

## F64 to F16 Policy

The official SiC model accepts 16-frame targets. The adapter uses four deterministic non-overlapping windows:

```text
[0:16], [16:32], [32:48], [48:64]
```

Every window uses a demonstration selected deterministically from official `3DPW_MC/train`. The same explicit window mask is applied to the demonstration input. Predictions replace only coordinates where `missing_mask=True`; all visible coordinates are restored exactly.

## Mask Policies

- `strict_official_mc`: accepts masks that remain constant inside every 16-frame window, hide exactly 6 or 10 joints (the official `drop_ratios_MC=[0.4,0.6]` support), and do not hide joints 0 or 16.
- `allow_ood_explicit`: runs any explicit F64 mask, but records why the mask is outside the official MC training distribution.

Temporal interruption results must be reported as an out-of-distribution SiC application unless a matching SiC training protocol is added.

## Direct Configuration

Edit `build_direct_run_config()` in `sars_adapter/run_completion.py`:

- `dry_run`: validate paths/query only or run inference.
- `query_path`: V2 `completion_query.pkl`; never point this to the private sidecar.
- `checkpoint_path`: frozen SiC checkpoint.
- `source_config`: training-compatible SiC YAML.
- `data_root`: official data root containing `3DPW_MC/train`.
- `output_path`: V2 `completion_result.pkl` returned to SARS-Inter.
- `device`: `cuda:0` or `cpu`.
- `mask_policy`: strict official MC support or explicit OOD research mode.
- `demonstration_seed`: deterministic train-only demonstration selection.
- `coordinate_transform_mode`: currently `identity_h36m17` only.

Run from the SiC repository root:

```powershell
<SIC_PYTHON> sars_adapter/run_completion.py
```

## Current Coordinate Limitation

The adapter records an explicit coordinate transform mode but currently implements identity H36M17 transfer only. Before formal cross-dataset reporting, audit axis direction, root convention, and scale between each SARS-Inter dataset and official 3DPW-MC. Any new normalization must be reversible and written to the result metadata.
