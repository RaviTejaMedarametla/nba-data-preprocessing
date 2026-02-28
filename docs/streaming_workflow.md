# Streaming Workflow

1. Load source and create deterministic dataset fingerprint.
2. Derive initial chunk and batch sizes from memory and compute constraints.
3. Process each chunk with incremental preprocessing and streaming-safe feature engineering.
4. Monitor process memory and telemetry before and after each chunk.
5. If memory budget is exceeded, shrink chunk size and retry (bounded retries).
6. Update online learning model via `partial_fit` on each accepted chunk.
7. Persist chunk-level metrics, quality checks, and benchmark outputs.
8. Emit aggregated report with throughput, latency, drift, and hardware telemetry.

## Resource-Aware Behavior

- **Adaptive chunk resizing** reduces OOM risk under constrained memory.
- **Compute-aware scaling** downshifts chunk and batch sizes when compute budget is capped.
- **Spill-to-disk mode** allows out-of-core continuation for edge devices.
- **Telemetry integration** captures CPU%, process RSS, system memory, and optional RAPL energy.
