# SARS Fair Coordinate Adapter Design

## Goal

Use a frozen Skeleton-in-Context checkpoint for objective completion comparisons on the SARS-Inter custom dataset and NW-UCLA without exposing evaluation data or modifying source datasets.

## Isolation Boundary

- SiC reads only `completion_query.pkl`, the frozen checkpoint, its source config, and the official `3DPW_MC/train` demonstration bank.
- SiC never reads the evaluation sidecar, labels, clean skeletons, recognition scores, or source dataset.
- The adapter writes a new `completion_result.pkl`; it never overwrites the query, checkpoint, source dataset, or prior result.
- SARS-Inter remains responsible for import, exact observed-joint restoration, recognition, statistics, and visualization.

## Canonical Transform

The SARS project H36M17 order is left-leg-first and right-arm-first. SiC/MotionBERT uses right-leg-first and left-arm-first. The forward permutation is:

```text
sic_joint[j] = project_joint[[0,4,5,6,1,2,3,7,8,9,10,14,15,16,11,12,13][j]]
```

The permutation is self-inverse. The project coordinate contract is `(x_lateral, y_depth, z_height)`. The fixed right-handed rotation into SiC Y-up space is:

```text
(x, y, z) -> (x, z, -y)
```

The inverse is `(x', y', z') -> (x', -z', y')`.

## Root And Scale Alignment

For each F64 sample, root and scale are estimated once and shared by all four F16 windows.

- Root trajectory uses visible pelvis coordinates. Missing pelvis frames use the visible two-hip midpoint. Remaining gaps are linearly interpolated from available root anchors, with nearest-anchor extension at sequence boundaries.
- Sample scale is the median length of H36M bones whose two endpoints are visible. It is computed over the complete F64 sample.
- A single deterministic `3DPW_MC/train` demonstration is selected per sample and reused for all windows.
- Reference scale is computed from the same visible-bone support after mapping the F64 mask to the repeated F16 demonstration.
- The query is translated and scaled into the selected demonstration frame. Missing coordinates are reset to zero after transformation.
- Model output is inverse-transformed back to project order and axes. Only missing coordinates are accepted; observed project coordinates are restored exactly.

The adapter fails explicitly when no root anchor or insufficient visible bone support exists. It does not use dataset-specific constants, clean poses, labels, interpolation-based joint completion, smoothing, or bone-length correction.

## Window Policy

F64 is split into `[0:16]`, `[16:32]`, `[32:48]`, and `[48:64]`. All windows share one demonstration, one sample scale, one root policy, one joint permutation, and one axis rotation. Non-overlapping inference is retained to match the current official SiC input length. Boundary position and velocity discontinuities are measured and reported, not silently smoothed.

## Provenance

The result metadata records:

- transform mode and version;
- joint permutation and axis matrix;
- sample/reference scale and visible-bone count;
- root-anchor policy and root-trajectory hashes;
- prompt identity, prompt file hash, and selection policy;
- window boundaries and boundary discontinuity diagnostics;
- checkpoint SHA256, repository commit/worktree identity, runtime, and exact observed-restoration error.

## Acceptance

- Forward/inverse round-trip error is below `1e-6` for visible synthetic inputs.
- Joint and mask permutation is exact.
- Missing coordinates remain zero before model inference.
- Four windows use one prompt identity and one sample scale.
- No private evaluation field is available to SiC.
- Imported observed coordinates are bitwise identical to the query source.
- Custom and NW-UCLA use the same checkpoint and transform mode.
- Real smoke outputs are finite, sample-ID aligned, independently stored, recognized, and visualized.

