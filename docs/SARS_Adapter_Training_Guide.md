# SiC Subset Training Guide for SARS-Inter Evaluation

## Purpose

This adapter trains Skeleton-in-Context (SiC) only from selected official skeleton tasks. It does not consume HAR labels, recognition scores, Utility Bank records, or target-dataset clean skeletons. The resulting checkpoint is therefore independent of the SARS-Inter recognizer and can be reused for every downstream HAR model.

For the SARS-Inter completion comparison, the recommended task is code-level `MC`, which corresponds to **Joint Completion (JC)** in the paper. This locally trained artifact must be described as a **locally MC-trained SiC baseline**, or a **subset-trained MC-only SiC baseline** when a numeric sample limit is used. It is not an official pretrained checkpoint.

## Why MC

| Code task | Official purpose | Fit for this project |
| --- | --- | --- |
| `PE` | 2D-to-3D pose estimation | Does not learn missing-joint completion. |
| `MP` | Predict future motion from observed history | Models future frames rather than spatial joint loss. |
| `MC` | Joint completion with 40% or 60% joints missing through a 16-frame clip | Closest official supervision to sustained joint occlusion. |
| `FPE` | Estimate future 3D poses from future 2D observations | Does not match explicit 3D missing-joint completion. |

The official MC loader uses ratios 0.4 and 0.6, which become 6 or 10 masked joints. It samples only joint indices 1 through 15, excluding indices 0 and 16, and sets the same joints to zero across all 16 frames. This is the closest official training distribution, but it is not identical to every SARS-Inter mask. Masks with changing joint sets, partial-window temporal boundaries, or unsupported joint counts remain out-of-distribution and must keep their adapter diagnostics.

## Isolation Boundary

Keep these resources separate:

- the upstream SiC source repository;
- a dedicated Python 3.7 environment with the upstream dependencies;
- an external data/checkpoint root;
- SARS-Inter query/result exchange artifacts.

Do not copy SiC dependencies into SARS-Inter or OmniControl.

## Official Data

For MC-only training, the data root only needs:

```text
<SIC_DATA_ROOT>/
  3DPW_MC/train, 3DPW_MC/test
  support_data/
```

When other tasks are enabled, their official directories are required as well. Training queries and demonstrations are selected only from official train directories. Test directories are never truncated by the subset selector.

## Direct Configuration

Edit `build_direct_run_config()` in `sars_adapter/train_subset.py`:

- `dry_run`: `True` validates and prints the plan; `False` starts GPU training.
- `data_root`: official ready-to-use data root.
- `checkpoint_root`: output root for checkpoints and manifests.
- `run_name`: unique run name; supports `{timestamp}`.
- `tasks`: non-empty ordered subset of `PE, MP, MC, FPE`; use `['MC']` for this comparison.
- `train_sample_limits`: define exactly the active tasks. `None` uses all train files; a positive integer selects a deterministic subset.
- `subset_seed`: controls the stable filename selection.
- `epochs`: training epochs.
- `batch_size`: physical single-GPU batch size.
- `test_batch_size`: batch size for later official evaluation.
- `num_workers`: DataLoader workers; start from `0` on Windows. Values such as `2` or `4` are allowed for formal training only with `persistent_workers=False`.
- `persistent_workers`: must remain `False` for strict resumable training. Workers are then recreated each epoch from the restored main Torch RNG; persistent worker-local RNG state cannot be captured in the checkpoint.
- `no_eval`: keep `True` during the subset run, then evaluate the final checkpoint separately.
- `seed`: model and training random seed.

Every real run saves `effective_config.yaml` and `training_subset_manifest.json` beside the checkpoint. The manifest records `task_scope=single_task` or `multi_task` and the exact selected files.
The effective configuration also materializes the resolved `data` section so that the checkpoint can be reloaded without relying on training-time mutation.

## Recommended Sequence

1. Set `tasks=['MC']`, `train_sample_limits={'MC': 64}`, one epoch, batch size 8, and run a dry run.
2. Run the same configuration as a one-epoch GPU smoke test.
3. Inspect elapsed time, peak memory, loss finiteness, and checkpoint round trip.
4. Freeze an MC pilot size, such as 4000 samples, or set `{'MC': None}` to use all official MC train files.
5. Train with a fixed seed and record the final checkpoint SHA256.
6. Evaluate the checkpoint only on official MC test data.
7. Use one frozen checkpoint for both the self-collected dataset and NW-UCLA.

Run from the SiC repository root:

```powershell
<SIC_PYTHON> sars_adapter/train_subset.py
```

To run the upstream evaluation after training:

```powershell
<SIC_PYTHON> train.py --config <RUN>/effective_config.yaml --checkpoint <RUN> --evaluate <RUN>/latest_epoch.bin
```

Formal 120-epoch training should start only after the one-epoch smoke run establishes the local runtime and a safe batch size.

With `no_eval=True`, training writes `latest_epoch.bin` and skips repeated evaluation over the complete MC test set. Evaluate it once after training. With `no_eval=False`, the runner also writes `best_epoch_MC.bin`; this is more expensive because official MC evaluation runs repeatedly during training. A single-task run also writes `best_epoch_all.bin`, which is numerically identical to the best MC selection and is retained only for filename compatibility.

Use the run-specific `effective_config.yaml`, rather than the repository default YAML, for evaluation and downstream completion. This keeps the MC-only task scope and exact training configuration bound to the checkpoint provenance. Formal completion must set `checkpoint_identity_policy=require_mc_only`; the adapter then compares the effective-config SHA256 stored in the checkpoint and rejects legacy or multi-task weights. If training is interrupted, resume from the same run directory with `train.py --config <RUN>/effective_config.yaml --checkpoint <RUN>`; the upstream loop detects `<RUN>/latest_epoch.bin`.

The one-epoch smoke checkpoint validates the runtime only. It is not suitable for reporting final completion accuracy.
