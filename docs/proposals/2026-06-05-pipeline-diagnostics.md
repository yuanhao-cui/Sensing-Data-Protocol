---
status: accepted
created: 2026-06-05
scope: pipeline-diagnostics
---

# Pipeline Diagnostics

## Goal

Add a small offline diagnostic tool that helps reject obviously bad CSI processing pipelines before full model training. The tool compares raw and processed CSI with fixed visual panels and simple proxy metrics, without starting services or using GPU by default.

## Acceptance Criteria

| Scenario | Expected behavior |
|---|---|
| User calls the diagnostic API with raw CSI and processed CSI of matching shape | It writes a contact-sheet PNG and a metrics CSV to the requested output directory. |
| User enables Doppler panels | The PNG includes raw and processed Doppler spectrogram summaries computed on CPU. |
| A deliberately over-smoothed pipeline removes motion-band energy from synthetic CSI | The metrics show lower motion-band energy ratio and lower dynamic energy than the original signal. |
| Inputs have incompatible shapes or unsupported dimensions | The tool raises a clear `ValueError` without writing partial outputs. |
| Tests run in CI/local pytest | Test artifacts are written only to pytest temporary directories; no service starts and no GPU is required. |

## Non-Goals

| Item | Reason |
|---|---|
| Full model training or final accuracy prediction | This tool is an early filter, not a replacement for task validation. |
| Web dashboard or long-running service | Offline files are simpler and easier to verify. |
| Trajectory-specific AoA/ToF visualizations | This first version targets dynamic recognition diagnostics. |
| Automatic dataset download | Diagnostics should run on already available arrays/samples. |

## Verification Plan

| Check | Command |
|---|---|
| Visualization/diagnostic unit tests | `.venv/bin/python -m pytest tests/test_visualization.py tests/test_pipeline_diagnostics.py -q` |
| Full existing visualization regression | `.venv/bin/python -m pytest tests/test_visualization.py -q` |
