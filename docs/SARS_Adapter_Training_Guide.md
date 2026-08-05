# SiC Subset Training Guide for SARS-Inter Evaluation

## Purpose

This adapter trains Skeleton-in-Context (SiC) only from its official PE, MP, MC, and FPE skeleton datasets. It does not consume HAR labels, recognition scores, Utility Bank records, or target-dataset clean skeletons. The resulting checkpoint is therefore independent of the SARS-Inter recognizer and can be reused for every downstream HAR model.

This locally trained artifact must be described as a **subset-trained SiC baseline**, not as an official pretrained checkpoint.

## Isolation Boundary

Keep these resources separate:

- the upstream SiC source repository;
- a dedicated Python 3.7 environment with the upstream dependencies;
- an external data/checkpoint root;
- SARS-Inter query/result exchange artifacts.

Do not copy SiC dependencies into SARS-Inter or OmniControl.

## Official Data

The data root must contain:

```text
<SIC_DATA_ROOT>/
  H36M/train, H36M/test
  AMASS/train, AMASS/test
  3DPW_MC/train, 3DPW_MC/test
  H36M_FPE/train, H36M_FPE/test
  source_data/H36M.pkl
  support_data/
```

Training queries and demonstrations are selected only from these official train directories. Test directories are never truncated by the subset selector.

## Direct Configuration

Edit `build_direct_run_config()` in `sars_adapter/train_subset.py`:

- `dry_run`: `True` validates and prints the plan; `False` starts GPU training.
- `data_root`: official ready-to-use data root.
- `checkpoint_root`: output root for checkpoints and manifests.
- `run_name`: unique run name; supports `{timestamp}`.
- `tasks`: must remain `PE, MP, MC, FPE`.
- `train_sample_limits`: deterministic query/prompt count per official training task.
- `subset_seed`: controls the stable filename selection.
- `epochs`: training epochs.
- `batch_size`: physical single-GPU batch size.
- `test_batch_size`: batch size for later official evaluation.
- `num_workers`: DataLoader workers; start from `0` on Windows.
- `no_eval`: keep `True` during the subset run, then evaluate the final checkpoint separately.
- `seed`: model and training random seed.

Every real run saves `effective_config.yaml` and `training_subset_manifest.json` beside the checkpoint.
The effective configuration also materializes the resolved `data` section so that the checkpoint can be reloaded without relying on training-time mutation.

## Recommended Sequence

1. Dry run with 64 samples per task, one epoch, batch size 8.
2. Run the same configuration as a one-epoch GPU smoke test.
3. Inspect elapsed time, peak memory, loss finiteness, and checkpoint round trip.
4. Freeze a pilot subset size, such as 1000 samples per task.
5. Train with a fixed seed and record the final checkpoint SHA256.
6. Evaluate the checkpoint on the official test tasks.
7. Use one frozen checkpoint for both the self-collected dataset and NW-UCLA.

Run from the SiC repository root:

```powershell
<SIC_PYTHON> sars_adapter/train_subset.py
```

To run the upstream evaluation after training:

```powershell
<SIC_PYTHON> train.py --config <RUN>/effective_config.yaml --evaluate <RUN>/latest_epoch.bin
```

Formal 120-epoch training should start only after the one-epoch smoke run establishes the local runtime and a safe batch size.

The one-epoch smoke checkpoint validates the runtime only. It is not suitable for reporting final completion accuracy.
