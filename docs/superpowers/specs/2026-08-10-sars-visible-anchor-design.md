# SARS Visible-Anchor Coordinate Adapter Design

## Goal

Allow the existing formal SARS H36M17 coordinate adapter to process masks such
as `bottom`, where pelvis and both hips are missing for all F64 frames, without
changing the explicit missing mask or reading clean/evaluation-only data.

## Scope

- Modify only the Skeleton-in-Context SARS adapter, tests, and adapter guides.
- Keep the existing public transform-mode identifier so current SARS-Inter run
  plans and direct-run configuration remain compatible.
- Replace the old root-only implementation directly; do not keep selectable
  legacy and new implementations.
- Do not modify the SiC network, checkpoint, prompt selection, SARS-Inter
  recognition, export/import, or evaluation behavior.

## Coordinate Contract

The adapter selects one deterministic semantic anchor policy per F64 sample:

1. pelvis with frame-level two-hip midpoint fallback;
2. center torso joint 7;
3. upper torso joint 8;
4. shoulder midpoint from joints 11 and 14;
5. centroid of a fixed, sorted set of at least three observable joints.

The query anchor uses only visible coordinates from `masked_keypoint` under the
explicit `missing_mask`. Missing anchor frames are interpolated, with nearest
observations extended at sequence boundaries. The prompt anchor uses the exact
same semantic joints from the train-only demonstration input. One F64 anchor
trajectory and one scale ratio are shared by all four non-overlapping F16
windows.

The adapter never fills a missing query joint before model inference. The same
explicit mask is applied to query and demonstration inputs. After inverse
transformation, only coordinates marked missing are accepted; all observed
project-space coordinates are restored exactly.

## Safety And Provenance

- No clean query skeleton, class label, HAR result, or test-derived prompt is
  available to anchor selection.
- Fully unobserved samples and samples with insufficient visible bone support
  still fail explicitly.
- Per-sample metadata records anchor mode, semantic joint IDs, observed and
  interpolated support, query/prompt anchor hashes, scale ratio, and exact
  observed-restoration error.
- Official MC compatibility remains reported independently. Enabling the
  coordinate adapter does not relabel `bottom` as an in-distribution MC mask.

## Acceptance

- Project `bottom` joints 0-7 missing for all frames use upper-torso anchor and
  reach SiC inference without changing the mask.
- Inputs previously supported by the root policy retain numerically equivalent
  canonical coordinates and inverse transforms.
- Forward/inverse synthetic error is below `1e-6`.
- Fully invisible inputs remain rejected.
- SARS-Inter source code is unchanged.
