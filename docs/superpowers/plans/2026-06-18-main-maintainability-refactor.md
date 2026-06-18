# Main Maintainability Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve readability, extensibility, and maintainability of the main branch while preserving public behavior.

**Architecture:** Keep public entry points stable and extract responsibilities behind them. Lock behavior with focused regression tests, then split core pipeline orchestration, algorithm convenience dispatch, registry config parsing, and reader loading helpers.

**Tech Stack:** Python 3.9, pytest, NumPy, PyTorch, scikit-learn, matplotlib/seaborn.

---

### Task 1: Accepted RFC and Plan

**Files:**
- Move: `docs/proposals/2026-06-18-main-maintainability-refactor.md` -> `docs/rfcs/0001-main-maintainability-refactor.md`
- Modify: `docs/STATE.md`
- Create: `docs/superpowers/plans/2026-06-18-main-maintainability-refactor.md`

- [x] **Step 1: Mark the proposal accepted**

Update the promoted RFC status to `accepted` and record that the user approved the conservative refactor path.

- [x] **Step 2: Update the docs index**

Point `docs/STATE.md` at RFC 0001 and mark the implementation as in progress.

### Task 2: Regression Tests Before Refactor

**Files:**
- Modify: `tests/test_core_configuration.py`
- Modify: `tests/test_registry.py`
- Modify: `tests/test_readers.py`

- [ ] **Step 1: Add core orchestration regression tests**

Cover hyperparameter override precedence, output directory creation, and cache hit behavior without running real training.

- [ ] **Step 2: Add algorithm dispatch regression tests**

Cover custom algorithms that do not accept every convenience-wrapper kwarg and feature extraction option routing.

- [ ] **Step 3: Add reader loading regression tests**

Cover file discovery excluding truth files, list-vs-single data extension, format mismatch skipping, and invalid/empty directories.

- [ ] **Step 4: Run targeted tests**

Run `pytest tests/test_core_configuration.py tests/test_registry.py tests/test_readers.py -q` and confirm the tests pass against current behavior or fail only because they reveal a refactor seam.

### Task 3: Core Pipeline Refactor

**Files:**
- Modify: `src/wsdp/core.py`
- Test: `tests/test_core_configuration.py`

- [ ] **Step 1: Extract small data containers and helpers**

Add internal containers for resolved hyperparameters, preprocessed arrays, split arrays, and seed run results. Extract helpers for output path setup, config loading, cache loading/saving, dataloader creation, model creation, plotting, and pipeline record persistence.

- [ ] **Step 2: Keep `pipeline(...)` as the public orchestrator**

Preserve its signature and defaults while reducing it to high-level steps: resolve config, load/preprocess, loop seeds, persist aggregate record.

- [ ] **Step 3: Run targeted core tests**

Run `pytest tests/test_core_configuration.py tests/test_cli.py -q`.

### Task 4: Algorithm API and Registry Refactor

**Files:**
- Modify: `src/wsdp/algorithms/__init__.py`
- Modify: `src/wsdp/algorithms/registry.py`
- Test: `tests/test_registry.py`
- Test: `tests/test_algorithms.py`

- [ ] **Step 1: Centralize algorithm invocation**

Replace repeated wrapper logic with internal helpers that resolve methods, filter kwargs, and include optional arguments only when supported.

- [ ] **Step 2: Centralize registry constants and config parsing helpers**

Introduce a single valid-category source and helper functions for preset merging, nested params flattening, and config serialization.

- [ ] **Step 3: Run targeted algorithm tests**

Run `pytest tests/test_registry.py tests/test_algorithms.py -q`.

### Task 5: Reader Entry Refactor

**Files:**
- Modify: `src/wsdp/readers/__init__.py`
- Test: `tests/test_readers.py`

- [ ] **Step 1: Extract reader input validation and file collection**

Add helpers for validating input directories, collecting candidate files, and consuming worker results.

- [ ] **Step 2: Preserve public reader API**

Keep `get_reader_class`, `list_datasets`, `get_all_reader_metadata`, and `load_data` behavior unchanged.

- [ ] **Step 3: Run targeted reader tests**

Run `pytest tests/test_readers.py -q`.

### Task 6: Final Verification and Push

**Files:**
- Modify: `docs/STATE.md`

- [ ] **Step 1: Run full verification**

Run `pytest` in the worktree virtualenv and read the full output.

- [ ] **Step 2: Update docs state**

Mark RFC 0001 implementation complete in `docs/STATE.md`.

- [ ] **Step 3: Commit and push**

Create conventional commits with Problem/Decision/Change bodies and push `refactor/main-maintainability` to `origin`.
