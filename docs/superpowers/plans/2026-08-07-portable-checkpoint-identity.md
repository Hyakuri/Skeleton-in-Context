# Portable SiC Checkpoint Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow smoke inference to use a checkpoint path directly while formal cross-device inference uses an automatically generated, portable, cryptographically verified checkpoint identity manifest.

**Architecture:** A focused `checkpoint_identity` module owns schema validation, hashing, portable path resolution, and atomic bundle creation. Existing single and series completion runners resolve either `direct_path` or `identity_manifest` into the same canonical runtime fields, then retain the current runtime re-hash and training-identity checks.

**Tech Stack:** Python 3, PyTorch checkpoint metadata, JSON, SHA256, `unittest`.

## Global Constraints

- Formal inference defaults to `identity_manifest`; `direct_path` is an explicit smoke/diagnostic mode.
- SHA256 verification remains mandatory and is recomputed on every device before model loading.
- Manifests contain relative filenames only; no absolute local paths or evaluation data.
- MC-only `training_identity` and matching `effective_config.yaml` remain mandatory for formal inference.
- No dependency installation, environment modification, or GPU inference is part of this change.
- Existing explicit `checkpoint_path` and `source_config` configurations remain compatible when `checkpoint_source_mode=direct_path`.

---

### Task 1: Portable checkpoint identity contract and freeze utility

**Files:**
- Create: `sars_adapter/checkpoint_identity.py`
- Create: `sars_adapter/freeze_checkpoint.py`
- Test: `tests/test_sars_checkpoint_identity.py`

**Interfaces:**
- Produces: `freeze_checkpoint_bundle(config) -> dict`
- Produces: `load_checkpoint_identity_manifest(path, verify_files=True) -> dict`
- Produces: `resolve_checkpoint_source(config) -> dict` with canonical `checkpoint_path`, `source_config`, `checkpoint_sha256`, and `checkpoint_identity_manifest_path`.

- [ ] Write tests for portable relative paths, automatic hashes, MC-only identity, copied bundle verification, tamper rejection, and direct-path resolution.
- [ ] Run `python -m unittest tests.test_sars_checkpoint_identity -v` and confirm failures are caused by the missing module.
- [ ] Implement schema version 1, atomic JSON save, optional atomic checkpoint/config copy, and content verification.
- [ ] Re-run the focused tests until all pass.
- [ ] Run `python -m py_compile sars_adapter/checkpoint_identity.py sars_adapter/freeze_checkpoint.py` and `git diff --check`.

### Task 2: Completion runner dual-mode integration

**Files:**
- Modify: `sars_adapter/run_completion.py`
- Modify: `sars_adapter/run_completion_series.py`
- Test: `tests/test_sars_completion_runner.py`
- Test: `tests/test_sars_completion_series.py`

**Interfaces:**
- Consumes: `resolve_checkpoint_source(config)` from Task 1.
- Preserves: existing result package `checkpoint_identity` as the actual checkpoint SHA256.

- [ ] Write tests proving formal manifest resolution, direct-path smoke compatibility, plan/hash mismatch rejection, and manifest tamper rejection before model load.
- [ ] Run focused tests and confirm the new cases fail before production edits.
- [ ] Add `checkpoint_source_mode`, `checkpoint_identity_manifest_path`, and canonical resolver calls to both runners.
- [ ] Keep current runtime asset re-verification and add manifest identity to provenance without changing result data semantics.
- [ ] Re-run runner and series tests, then compile modified modules.

### Task 3: User documentation and regression verification

**Files:**
- Modify: `docs/SARS_Adapter_Inference_Guide.md`
- Modify: `docs/SARS_Adapter_Inference_Guide_CN.md`
- Modify: `docs/SARS_Fair_Completion_Pipeline_Guide.md`
- Modify: `docs/SARS_Fair_Completion_Pipeline_Guide_CN.md`

**Interfaces:**
- Documents: direct-path smoke workflow, checkpoint freeze workflow, cross-device bundle transfer, and formal manifest execution.

- [ ] Add matching English and Chinese instructions and direct-run parameter tables.
- [ ] Run all SiC adapter tests with the isolated `SkeletonInContext` environment.
- [ ] Run `git diff --check` and inspect the final scoped diff.

