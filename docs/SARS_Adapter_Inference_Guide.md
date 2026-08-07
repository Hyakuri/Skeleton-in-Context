# SARS-Inter Completion Adapter for Skeleton-in-Context

## Boundary

The external process reads only `completion_query.pkl`. SARS-Inter keeps `completion_evaluation_sidecar.pkl` private. The query contains masked H36M17/F64 coordinates, an explicit Boolean `missing_mask`, sample order, and coordinate metadata. It contains no HAR labels, recognition scores, or clean skeleton.

External exchange PKL files use Pickle protocol 4 so the Python 3.7 SiC environment can read artifacts written by a newer SARS-Inter environment.

## F64 to F16 Policy

The official SiC model accepts 16-frame targets. The adapter uses four deterministic non-overlapping windows:

```text
[0:16], [16:32], [32:48], [48:64]
```

Formal comparison mode selects one demonstration deterministically from official `3DPW_MC/train` for each F64 sample and reuses it for all four windows. The same explicit window mask is applied to the demonstration input. Predictions replace only coordinates where `missing_mask=True`; all visible coordinates are restored exactly.

The adapter reports position and velocity discontinuities at frames 16, 32, and 48. It does not smooth or interpolate generated joints, so SiC does not receive post-processing unavailable to other completion methods.

## Mask Policies

- `strict_official_mc`: accepts masks that remain constant inside every 16-frame window, hide exactly 6 or 10 joints (the official `drop_ratios_MC=[0.4,0.6]` support), and do not hide joints 0 or 16.
- `allow_ood_explicit`: runs any explicit F64 mask, but records why the mask is outside the official MC training distribution.

Temporal interruption results must be reported as an out-of-distribution SiC application unless a matching SiC training protocol is added.

## Checkpoint Source Modes

Two explicit modes share the same downstream inference path:

- `direct_path` is for smoke and diagnostic runs. Fill `checkpoint_path` and the matching `source_config`; the runner computes the checkpoint SHA256 automatically. A dry-run reports the computed hash without loading the model.
- `identity_manifest` is the formal default. It accepts only a portable `checkpoint_identity.json` with `checkpoint_identity_policy=require_mc_only`. The manifest binds an MC-only checkpoint, its exact `effective_config.yaml`, sizes, SHA256 values, training identity, and its own canonical hash.

Create the formal bundle by editing `build_direct_run_config()` in `sars_adapter/freeze_checkpoint.py`. First use `dry_run=True` to inspect the identities, then set `dry_run=False` to publish the bundle. A bundle has this portable layout:

```text
<bundle>/
  checkpoint_identity.json
  <selected-checkpoint>.bin
  effective_config.yaml
```

The manifest stores relative filenames only. Copy the whole directory to another computer and configure only the new `checkpoint_identity_manifest_path`; do not manually copy the old absolute paths or retype the SHA256. An existing different bundle is never replaced in place, even if an old config still contains `overwrite=True`; create a new output directory instead.

## Direct Configuration

Edit `build_direct_run_config()` in `sars_adapter/run_completion.py`:

- `dry_run`: validate paths/query only or run inference.
- `query_path`: V2 `completion_query.pkl`; never point this to the private sidecar.
- `checkpoint_source_mode`: `identity_manifest` for formal runs or `direct_path` for smoke runs.
- `checkpoint_identity_manifest_path`: required only by formal mode; points to the copied bundle's `checkpoint_identity.json`.
- `checkpoint_path`: required only by `direct_path`; its SHA256 is calculated automatically.
- `source_config`: required only by `direct_path`; it must be the run-specific `effective_config.yaml` matching the checkpoint.
- `checkpoint_identity_policy`: formal manifest mode is fixed to `require_mc_only`. `allow_legacy` is accepted only with `direct_path` for old smoke checkpoints and must not be used for paper results.
- `data_root`: official data root containing `3DPW_MC/train`.
- `output_path`: V2 `completion_result.pkl` returned to SARS-Inter.
- `device`: `cuda:0` or `cpu`.
- `mask_policy`: strict official MC support or explicit OOD research mode.
- `demonstration_seed`: deterministic train-only demonstration selection.
- `demonstration_selection_policy`: use `per_sample_fixed` for formal comparison; all four windows share one train demonstration.
- `coordinate_transform_mode`: use `project_h36m17_prompt_aligned_v1` for formal SARS-Inter comparison. `identity_h36m17` remains available only for historical reproduction.

The two options are intentionally coupled. Formal coordinate mode requires `per_sample_fixed`; legacy identity mode requires `per_window`. A mismatched pair is rejected instead of silently changing historical behavior.

Run from the SiC repository root:

```powershell
<SIC_PYTHON> sars_adapter/run_completion.py
```

## Formal Coordinate Adapter

`project_h36m17_prompt_aligned_v1` applies the same deterministic policy to the custom dataset and NW-UCLA:

The query coordinate contract must explicitly declare `joint_order=h36m17_sars_inter_project_order` and `axis_order=[x_lateral,y_depth,z_height]`. This prevents a standard-order H36M17 package from being permuted a second time.

1. Swap the project left-leg/right-leg and right-arm/left-arm groups into the SiC/MotionBERT H36M17 order.
2. Rotate project Z-up coordinates with `(x, y_depth, z_height) -> (x, z_height, -y_depth)`.
3. Estimate one F64 root trajectory and one visible-bone scale using only observed query coordinates.
4. Align the query to the selected train demonstration, run four F16 windows, and inverse-transform the output.
5. Restore every observed project-space coordinate exactly.

The result records the permutation, axis matrix, root hashes, sample/reference scales, prompt identity/hash, prompt-pool manifest hash/count, source-config hash, checkpoint training identity, missing-joint boundary diagnostics, repository identity, and checkpoint SHA256. Prompt selection uses a stable hash of `masked_keypoint + missing_mask`; it never parses sample IDs, labels, or dataset names.

## Custom And NW-UCLA Runs

Run `sars_adapter/run_completion.py` once per query. Use the same frozen checkpoint, `source_config`, `data_root`, `demonstration_seed`, `mask_policy`, selection policy, and coordinate mode for both datasets. Change only `query_path` and `output_path`.

Keep each result in a new method-specific run directory. Do not expose the parent directory containing `completion_evaluation_sidecar.pkl` to SiC; pass only the query file path. Import and evaluate each result through SARS-Inter with its dataset-specific frozen recognizer checkpoint. Report the two datasets separately.
