# SARS Fair Coordinate Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reversible, leakage-free project-H36M17 to SiC coordinate adapter and validate it with the frozen 4,000-sample checkpoint on custom and NW-UCLA smoke queries.

**Architecture:** A focused `coordinate_adapter.py` owns joint, axis, root, and scale conversion. `inference.py` applies one sample-level transform and one train-only prompt across four F16 windows. `run_completion.py` exposes the mode and records complete provenance while preserving the existing identity mode.

**Tech Stack:** Python 3.7, NumPy, PyTorch, unittest, pickle protocol 4.

## Global Constraints

- Do not read the evaluation sidecar, labels, clean skeletons, recognition scores, or source dataset from SiC.
- Do not overwrite source queries, checkpoints, datasets, or historical outputs.
- Do not install or modify dependencies.
- Keep `identity_h36m17` backward compatible.
- Use Chinese comments and bilingual user guides.
- Run SiC tests/inference only with `SkeletonInContext`; run SARS import/recognition only with `SARS-InterV2`.

---

### Task 1: Reversible Coordinate Adapter

**Files:**
- Create: `sars_adapter/coordinate_adapter.py`
- Modify: `tests/test_sars_completion_adapter.py`

**Interfaces:**
- Produces: `prepare_project_h36m17_sample(masked, missing, prompt_input) -> (canonical, canonical_mask, state, metadata)`.
- Produces: `inverse_project_h36m17_window(generated, state, start, end) -> ndarray`.

- [ ] Add failing tests for joint/mask permutation, axis round-trip, shared F64 root/scale, zeroed missing coordinates, and missing-root rejection.
- [ ] Run the focused tests and verify failures are caused by the missing module/API.
- [ ] Implement the minimal reversible adapter with visible-only root/scale estimation.
- [ ] Run the focused tests and verify they pass.

### Task 2: One-Prompt F64 Inference

**Files:**
- Modify: `sars_adapter/inference.py`
- Modify: `sars_adapter/run_completion.py`
- Modify: `tests/test_sars_completion_adapter.py`

**Interfaces:**
- Extends: `build_train_prompt_provider(..., selection_policy='per_window')`.
- Extends: `complete_f64_with_model(..., coordinate_transform_mode='identity_h36m17')`.

- [ ] Add failing tests that formal mode uses one prompt identity, one scale, transformed mask semantics, exact observed restoration, and transform metadata.
- [ ] Run tests and verify the new expectations fail under identity-only code.
- [ ] Add `project_h36m17_prompt_aligned_v1` and `per_sample_fixed` while preserving identity behavior.
- [ ] Add deterministic position/velocity boundary diagnostics without output smoothing.
- [ ] Run all SiC adapter tests.

### Task 3: Direct Configuration And Guides

**Files:**
- Modify: `sars_adapter/run_completion.py`
- Modify: `docs/SARS_Adapter_Inference_Guide.md`
- Modify: `docs/SARS_Adapter_Inference_Guide_CN.md`

**Interfaces:**
- Configures the formal transform, frozen checkpoint, query path, output path, device, and prompt policy from `build_direct_run_config()`.

- [ ] Make formal mode the documented recommendation without removing identity compatibility.
- [ ] Document the physical isolation, no-leakage contract, two-dataset command sequence, and output interpretation.
- [ ] Run `py_compile`, unittests, and `git diff --check`.

### Task 4: Real Dual-Dataset Smoke

**Files:**
- Generate only under a new external smoke run directory; do not commit outputs.

**Interfaces:**
- Consumes the existing custom and NW-UCLA V2 query files and the frozen subset checkpoint.
- Produces new formal-adapter completion results, imported datasets, full/selected recognition analyses, and comparison videos.

- [ ] Run SiC inference for one custom and one NW-UCLA query.
- [ ] Validate hashes, sample order, finite arrays, transform metadata, boundary diagnostics, and observed restoration.
- [ ] Import with SARS-InterV2 into new output paths.
- [ ] Run full/selected recognition with frozen dataset-specific recognizers.
- [ ] Generate and inspect comparison videos.
- [ ] Write a local smoke summary without committing local paths or data.

### Task 5: Publish SiC Branch

**Files:**
- Commit only source, tests, and documentation.

**Interfaces:**
- Publishes `codex/sic-fair-completion-pipeline` based on `sars-inter-external-adapter`.

- [ ] Re-fetch and confirm no divergence or conflict.
- [ ] Verify intended diff and clean status.
- [ ] Commit and push the branch.
- [ ] Open a PR against `sars-inter-external-adapter`.
- [ ] Add a three-section PR comment: problem summary, solution, actual run report.

