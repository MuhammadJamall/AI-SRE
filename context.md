# Aether Project Context

This document is a handoff for an external LLM or developer working in this repository. It describes the code and data that exist now, not just the broader product vision. Verify the current source before relying on this note if the repository changes.

## Project Summary

Aether is an AI SRE Copilot prototype. The README describes the intended product as detecting anomalies in server metrics and HDFS logs, correlating them into incidents, explaining them with a pretrained LLM, and recommending troubleshooting steps. It explicitly says recommendations only: Aether must not execute remediation.

**Implemented today:** Module 1 ingestion for SMD and HDFS, and Module 2 metric anomaly detection using a per-machine LSTM autoencoder. HDFS event sequences are independently verified against the supplied official event-trace file. The correlation, LLM explanation, log anomaly modeling, preprocessing, and remediation packages are empty placeholders; the README's full product flow is not implemented yet.

The code is a Python package-style repository run from its root with `python -m ...`. There is no `pyproject.toml` or setup script in the current tree.

## Repository Layout

```text
configs/default.yaml                 Runtime configuration
src/common/                          Shared schemas, config, I/O, logging, validation
src/ingestion/smd.py                 SMD raw-to-metrics adapter
src/ingestion/hdfs.py                HDFS raw-to-logs-and-traces adapter
src/ingestion/pipeline.py            Module 1 command-line entry point
src/anomaly_metrics/data.py          Parquet loading, windows, scaling, splits
src/anomaly_metrics/model.py         LSTM autoencoder architecture
src/anomaly_metrics/train.py         Deterministic training and reconstruction errors
src/anomaly_metrics/evaluate.py      Confusion metrics, point adjustment, oracle F1
src/anomaly_metrics/plots.py         Headless diagnostic plots
src/anomaly_metrics/pipeline.py      Single-machine training/scoring/artifact generation
src/anomaly_metrics/run.py           Module 2 command-line entry point
src/anomaly_logs/                    Placeholder package; no implementation
src/correlation/                     Placeholder package; no implementation
src/llm/                             Placeholder package; no implementation
src/preprocessing/                   Placeholder package; no implementation
src/remediation/                     Placeholder package; no implementation
scripts/verify_hdfs_vs_event_traces.py  Full-dataset HDFS sequence comparison
 tests/                              Synthetic, validation, config, and opt-in real-data tests
```
Each machine is trained independently. For a machine's chronological training array, the last `val_fraction` (20% by default) is held out for validation; the scaler is fit only on the earlier fit portion. Training rows are treated as normal. The held-out training tail is assumed normal for threshold calibration; the code does not independently verify that assumption. Validation data determines the global reconstruction-error percentile threshold and per-feature thresholds. Test labels are used for reported evaluation metrics only, not model fitting or threshold choice.

The leading space in the last tree entry is typographical only; the actual directory is `tests/`.

## Environment and Configuration

Dependencies are listed in `requirements.txt`: pandas, NumPy, PyArrow, PyYAML, pytest, PyTorch, and Matplotlib. The exercised environment is macOS, Python 3.12.14, with a local `.venv`. Install and test from repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -v
```

`configs/default.yaml` is loaded by `src.common.config.load_config`. Resolution order is explicit `--config` path, then `$AETHER_CONFIG`, then `configs/default.yaml`. `resolve()` leaves absolute paths unchanged and resolves relative paths against `PROJECT_ROOT`, which is derived from `src/common/config.py`.

Current configuration:

- Project seed: `42`.
- Raw data root: `data/raw`; interim: `data/interim`; processed: `data/processed`.
- SMD root: `data/raw/ServerMachineDataset`.
- HDFS root: `data/raw/hdfs`, train ratio `0.7`, batch size `200000`.
- Module 2 machine subset: `machine-1-1`, `machine-2-1`, `machine-3-1`.
- Module 2: window 60, train stride 5, validation fraction 0.2, hidden size 64, latent size 16, one LSTM layer, batch size 64, 30 epochs, learning rate 0.001, patience 5, threshold percentile 99.5, CPU by default.
- Module 2 models: `models/anomaly_metrics`; results: `data/processed/anomaly_metrics`.

`.gitignore` excludes `data/raw/`, `data/processed/`, `models/`, virtualenv/cache directories, and bytecode. Generated reports, Parquet files, checkpoints, scalers, and plots therefore exist locally but are normally not tracked. Preserve this behavior unless explicitly asked to change data/model tracking.

## Shared Contracts

`src/common/schema.py` is the source for required columns:

- Metrics: `timestamp`, `entity_id`, `label`, `split`, plus numeric metric columns.
- Logs: `timestamp`, `log_message`, `event_id`, `trace_id`, `label`, `split`.
- Traces: `trace_id`, `start_time`, `event_sequence`, `label`, `split`.
- Valid labels are integers `{0, 1}`; valid splits are `{"train", "test"}`.

`src/common/validation.py` defines `ValidationError`, metric cleaning, and validators:

- `metric_columns(df)` excludes the four metric-required columns; use it only with a metrics frame.
- `clean_metrics(df)` replaces infinities with missing values, drops duplicate `(entity_id, timestamp)` rows keeping the first, sorts by entity/time, forward- then backward-fills metric columns within entity, then fills any remaining gaps with zero. It returns `(cleaned_df, report)`.
- `validate_metrics` checks required fields, non-empty frame, datetime timestamps, numeric finite metrics, binary labels, allowed splits, unique entity/timestamp keys, monotonic timestamps per entity, and strict train-before-test chronology. It returns summary counts and anomaly rate.
- `validate_logs` checks required fields, timestamps, non-null messages/event/trace IDs, labels/splits, and a single label and split per trace.
- `validate_traces` checks required fields, datetime `start_time`, unique trace IDs, nonempty sequences, and valid labels/splits.

`src/common/io.py` provides `save_json(data, path)` and `save_parquet(df, path)`. Both create parent directories. JSON is UTF-8, indented, newline-terminated; Parquet excludes the pandas index. `src/common/logging_utils.py` exposes `setup_logging(level)` and `get_logger(name)`.

## Raw Data Present

The real datasets are already local. Do not re-download or relocate them without a reason:

- SMD: `data/raw/ServerMachineDataset/` with `train/`, `test/`, `test_label/`, `interpretation_label/`.
- HDFS: `data/raw/hdfs/` with `HDFS.log`, `anomaly_label.csv`, `HDFS.log_templates.csv`, `Event_traces.csv`, `Event_occurrence_matrix.csv`, and `HDFS.npz`.
- Other raw collections include the NAB time-series archive, five undocumented headerless cluster CSVs under `data/raw/archive (3)/`, and `data/raw/ci_cd_logs.csv`. They currently have no ingestion adapters in this project.

For a complete per-file dataset inventory, raw schemas, counts, and data caveats, read [`data/raw/LLM_CONTEXT.md`](data/raw/LLM_CONTEXT.md). SMD has a local MIT license at `data/raw/ServerMachineDataset/LICENSE`; licensing/redistribution terms for the other raw collections were not established in this repository.

## Module 1: Ingestion

### SMD

`src/ingestion/smd.py` reads each machine's headerless numeric matrices with pandas. A machine has 38 unlabeled dimensions in this dataset; the adapter names them `m00` through `m37`. There are 28 machines. `test_label` has one 0/1 label per test row; training rows are assigned label 0. Since source matrices have no timestamps, the adapter synthesizes one-minute timestamps from `2020-01-01`; each test series begins immediately after that machine's training series.

Train and test lengths need not match. The current raw data has 15 machines with exactly one extra test observation. Interpretation annotations are optional per-machine files; parser assumes 1-based row/dimension indices and stores them zero-based in `interpretation.json`.

`ingest_smd(raw_root, out_dir, machines=None)` loads all discovered machines unless a list is passed, cleans and validates the combined frame, then writes:

- `train.parquet`
- `test.parquet`
- `interpretation.json`
- `report.json`

Current real SMD ingestion facts: 1,416,825 total rows (708,405 train, 708,420 test), 28 entities, 38 metrics, 29,444 positive test labels. The report is `data/processed/smd/report.json`.

### HDFS

`src/ingestion/hdfs.py` streams `HDFS.log`; do not load the entire text file into memory. It can find required files either at the raw root or in `preprocessed/`. Labels are `anomaly_label.csv`; templates may be named `HDFS.log_templates.csv` (the current local name) or `HDFS_templates.csv` (legacy/test layout).

The line regex supports both timestamp/PID layouts observed or anticipated by the project:

```text
YYMMDD HHMMSS PID LEVEL component: content
YYMMDD HHMMSS YYMMDD HHMMSS LEVEL component: content
```

The timestamp comes from the first date/time pair; the third field is captured but ignored. `parse_timestamp` interprets the two-digit year as 2000 + YY. Log messages are matched against official templates; both `[*]` and `<*>` wildcard notation are supported. If no official template matches, event ID is `UNKNOWN` and reported.

Block IDs found in a line determine trace association. The log row is attached to the first block ID in the line; the trace sequence appends the event to every labeled block on that line. A trace's label comes from the block label map. Train/test split uses deterministic `zlib.crc32(trace_id) % 1000` against configured `train_ratio`, preventing the same block from spanning both splits. Logs are written in batches to avoid full-log memory use.

`ingest_hdfs(raw_root, out_dir, max_lines=None, batch_size=200000, train_ratio=0.7)` writes `logs.parquet`, `traces.parquet`, `templates.json`, and `report.json`. Current full-run facts: 11,175,629 log rows, 575,061 traces, 16,838 anomalous traces, 29 official templates, zero skipped lines, zero unlabeled-block lines, and zero unknown-template events. The report is `data/processed/hdfs/report.json`.

The verification-only script `scripts/verify_hdfs_vs_event_traces.py` compares each ingested event sequence against the included official `Event_traces.csv`. The last full comparison had 575,061 official traces, zero missing, zero sequence mismatches, and 100.0000% exact match. It expects a completed full ingest, not just `--max-lines` sample outputs.

### Module 1 CLI

```bash
python -m src.ingestion.pipeline smd
python -m src.ingestion.pipeline hdfs --max-lines 500000
python -m src.ingestion.pipeline hdfs
```

The CLI accepts `--config`, `--raw`, `--out`; `--max-lines` is HDFS-only. Defaults come from `configs/default.yaml`. Outputs are `data/processed/smd/` and `data/processed/hdfs/`. `--max-lines` overwrites the same HDFS output locations, so a later full run is required before using the complete report/sequence verifier.

## Module 2: Metric Anomaly Detection

### Data and model

`src/anomaly_metrics/data.py` loads the SMD train/test Parquet files, optionally filtering entities; extracts metric columns as float32 arrays; makes a chronological fit/validation split; builds memory-view sliding windows; and provides a small `StandardScaler`. The scaler replaces near-zero standard deviations with 1 and supports JSON save/load. Windows must be 2D input and at least as long as the window size.

`src/anomaly_metrics/model.py` implements `LSTMAutoencoder`: an encoder LSTM reduces a window to a latent vector, a linear bottleneck maps to latent size, and a decoder LSTM reconstructs the full window.

### Training and scoring

`src/anomaly_metrics/train.py` seeds Python, NumPy, and Torch; selects CPU by default; and uses MPS only when explicitly requested and available (otherwise warns and falls back to CPU). It samples training windows using configured stride, shuffles window indices deterministically, trains with Adam/MSE, clips gradient norm at 1, monitors validation loss, early-stops after configured patience, and restores the best validation state.

Each machine is trained independently. For a machine's chronological training array, the last `val_fraction` (20% by default) is held out for validation; the scaler is fit only on the earlier fit portion. Training rows are treated as normal. Validation data determines the global reconstruction-error percentile threshold and per-feature thresholds. Test labels are used for reported evaluation metrics only, not model fitting or threshold choice.

`reconstruction_errors` returns squared reconstruction errors per point and feature. A point uses the reconstructed last step of its ending window; the first `window - 1` points use the corresponding positions from the first window. The global raw point score is mean error across features. `anomaly_score = raw_error / (raw_error + threshold)`; `is_anomaly` is equivalent to `raw_error > threshold`, so scores are above 0.5 exactly when flagged.

Affected metrics are the top five feature error/validation-threshold ratios greater than 1. If no feature exceeds its own threshold, the list is empty (do not substitute the top feature regardless of threshold).

### Evaluation and artifacts

`src/anomaly_metrics/evaluate.py` reports confusion counts and point-wise precision/recall/F1. It also reports point-adjusted metrics, which mark an entire contiguous true-anomaly segment as detected if any point in it was detected; this is more lenient and must be interpreted alongside point-wise scores. `oracle_best_f1` sweeps score thresholds using test labels. It is explicitly an oracle diagnostic, not a deployable operating metric.

`src/anomaly_metrics/plots.py` writes non-interactive Matplotlib plots: score distributions (validation/test normal/test anomaly), error timeline with actual/predicted labels, and an example anomaly window with high-error metrics.

`run_machine(...)` in `src/anomaly_metrics/pipeline.py` performs one entity's scaling/training/scoring/evaluation, optionally saving model checkpoint, scaler, thresholds, metrics, and plots. The CLI in `src/anomaly_metrics/run.py` loads processed SMD splits, runs either configured machines, a requested list, or all machines, then writes:

- `models/anomaly_metrics/<machine>/model.pt`, `scaler.json`, `threshold.json`
- `data/processed/anomaly_metrics/<machine>/metrics.json` and plots
- `data/processed/anomaly_metrics/summary.json`
- `data/processed/anomaly_metrics/predictions.parquet`

CLI examples:

```bash
python -m src.anomaly_metrics.run
python -m src.anomaly_metrics.run --machines machine-1-1
python -m src.anomaly_metrics.run --all --epochs 20
python -m src.anomaly_metrics.run --device mps
python -m src.anomaly_metrics.run --no-plots
```

Current configured three-machine run completed on CPU. Latest reported aggregate numbers: micro F1 0.2578; macro precision 0.2874, recall 0.5623, F1 0.2006; macro point-adjusted F1 0.4918. Per-machine point-wise F1: machine-1-1 0.3622, machine-2-1 0.1674, machine-3-1 0.0721. Results are dataset/config-specific, not guaranteed general performance. Machine-2-1's point-adjusted F1 is much higher than its point-wise F1, illustrating why both must be shown.

## Test Suite and Current Verification

Run all tests with `.venv/bin/pytest -v`. Tests are grouped as follows:

- `tests/test_config.py`: default paths, explicit config precedence, environment config, relative/absolute path resolution.
- `tests/test_validation.py`: valid metric summary; NaN, infinity, duplicate entity/timestamp, and temporal leakage rejection.
- `tests/test_smd.py`: output splits, chronological machine boundaries, unequal lengths, interpretation index conversion, determinism, missing files/directories, bad values, label lengths.
- `tests/test_hdfs.py`: timestamp parsing, both line layouts, masked content stability, both template wildcard styles, invalid/missing files, flat and preprocessed layouts, ingestion outputs, batch-size consistency, deterministic split.
- `tests/test_anomaly_metrics.py`: scaler round-trip/constant columns, splits/windows, model shape, metric calculations, affected metrics, loss/inference, injected anomalies, reproducibility, and threshold independence from test data.
- `tests/test_real_data.py`: real SMD, 500k-line HDFS sample, opt-in full HDFS, and 2-epoch Module 2 smoke test.

Real-data tests are marked `real_data`; the full HDFS test requires `AETHER_FULL=1`. The last full suite result was **43 passed, 1 skipped** (the opt-in full HDFS test). The Module 2 smoke test passed. The full HDFS test and independent sequence comparison also passed in the prior Module 1 run.

Useful targeted commands:

```bash
.venv/bin/pytest -v -m real_data
.venv/bin/pytest -v -m real_data -k smoke
AETHER_FULL=1 .venv/bin/pytest -v -m real_data -k full
.venv/bin/python -m src.anomaly_metrics.run
```

## Current Boundaries and Cautions

- Do not claim that log anomaly detection, cross-signal incident correlation, LLM explanations, or remediation execution are implemented. Their package directories currently contain only `__init__.py` placeholders.
- Do not execute remediation; the product contract is recommendations only.
- Keep test labels out of training and threshold selection. The oracle-best-F1 calculation intentionally uses test labels only as a diagnostic and must not be presented as production performance.
- SMD timestamps are synthesized, not source timestamps. SMD interpretation range boundary semantics are not separately validated beyond the existing 1-based-to-zero-based conversion test.
- HDFS `max_lines` is a smoke-test mode and produces partial traces/labels. Do not compare its counts to full-dataset totals or use it for the official trace comparison.
- HDFS sequence verification only validates event sequence reconstruction, not every possible temporal or operational property.
- NAB has no point-label windows in the local raw folder; archive (3) is headerless and undocumented; CI/CD logs have statuses but no anomaly label. These are not currently part of either implemented adapter.
- Preserve raw files and avoid whole-file loading of `HDFS.log`.
- `README.md` documents Module 1 commands; this file has the broader current code and Module 2 context.
