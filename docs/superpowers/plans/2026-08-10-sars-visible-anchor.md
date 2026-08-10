# SARS Visible-Anchor Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the root-only SARS coordinate alignment with one deterministic paired semantic-anchor policy that supports the paper `bottom` mask.

**Architecture:** Build one F64 query/prompt anchor pair before windowing, retain the existing scale and joint/axis contracts, and reuse one inverse state for all four F16 windows. Keep the public transform mode stable and record the selected anchor in result metadata.

**Tech Stack:** Python 3.7-compatible syntax, NumPy, PyTorch test doubles, unittest.

## Global Constraints

- Use the isolated `SkeletonInContext` environment for SiC tests.
- Do not run real GPU inference without separate confirmation.
- Preserve the user's uncommitted direct-run path change.
- Do not modify SARS-Inter production code.
- Do not change the explicit mask or observed-joint restoration policy.

### Task 1: Regression tests

**Files:**
- Modify: `tests/test_sars_coordinate_adapter.py`
- Modify: `tests/test_sars_completion_adapter.py`
- Modify: `tests/test_sars_completion_series.py`

- [x] Add a project-bottom test that requires upper-torso anchor selection.
- [x] Add root-compatible numerical-equivalence and fully invisible rejection tests.
- [x] Add a full completion test that proves bottom root remains masked and observed joints restore exactly.
- [x] Run the focused tests and confirm the new bottom tests fail for the current root-only implementation.

### Task 2: Paired anchor implementation

**Files:**
- Modify: `sars_adapter/coordinate_adapter.py`

- [x] Implement deterministic anchor candidates and shared F64 interpolation.
- [x] Store generic query/prompt anchor state and metadata.
- [x] Keep the existing transform mode identifier and inverse API.
- [x] Run focused tests and confirm all new cases pass.

### Task 3: Audit and documentation

**Files:**
- Modify: `sars_adapter/run_completion.py`
- Modify: `docs/SARS_Adapter_Inference_Guide.md`
- Modify: `docs/SARS_Adapter_Inference_Guide_CN.md`
- Modify: `docs/SARS_Fair_Completion_Pipeline_Guide.md`
- Modify: `docs/SARS_Fair_Completion_Pipeline_Guide_CN.md`

- [x] Capture Git identity stderr without changing identity bytes or failure handling.
- [x] Document the anchor hierarchy, OOD boundary, and exact restoration contract.
- [x] Run py_compile, the complete SARS adapter unittest set, and `git diff --check`.
- [x] Review that only SiC adapter-related files are staged.
