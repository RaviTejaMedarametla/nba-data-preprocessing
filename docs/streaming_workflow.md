# Streaming Workflow and Runtime Analysis

## End-to-End Workflow

1. Load source data and create a deterministic fingerprint.
2. Derive initial chunk and batch sizes from configured memory and compute limits.
3. Preprocess each chunk and build streaming-safe features.
4. Validate chunk outputs and collect telemetry snapshots.
5. If memory limits are exceeded, reduce chunk size and retry (bounded).
6. Update online learning state with `partial_fit` per accepted chunk.
7. Write chunk-level logs and aggregate benchmark artifacts.
8. Emit a final report with quality checks, performance summaries, and drift metrics.

## Runtime Behavior Under Constraints

- **Adaptive chunk resizing** lowers OOM risk at the cost of additional orchestration latency.
- **Compute-aware scaling** reduces chunk/batch sizes when compute budget is capped.
- **Spill-to-disk mode** preserves progress in low-memory environments with extra I/O latency.
- **Telemetry integration** captures CPU load, RSS memory, system memory pressure, and optional RAPL energy.

## Layer-Wise Latency and Memory Decomposition

The pipeline is organized into four major operators per chunk:

1. Clean and normalize records.
2. Build and update engineered features.
3. Drop high-correlation features and encode/scale inputs.
4. Run model update/evaluation.

Interpretation guidance:

- If preprocessing dominates latency, investigate parsing and categorical handling.
- If feature encoding dominates memory, inspect category cardinality.
- If model update latency grows with chunk size, retune chunk/batch parameters.

## Bandwidth and Utilization Notes

- Input bandwidth is driven by CSV read rate and storage throughput.
- Memory bandwidth pressure increases during one-hot encoding and concatenation.
- CPU utilization can spike during bootstrap/permutation statistics in benchmarks.
- Throughput should be evaluated jointly with peak RSS to avoid unstable operating points.

## Precision and Quantization Trade-Offs

Current implementation uses floating-point operations in pandas/numpy/scikit-learn defaults.

- Lower-precision representations can reduce memory footprint.
- Precision reduction may change model quality and drift estimates.
- Quantization is not enabled by default and should be validated per deployment target.

## Assumptions

- Input schema is stable across chunks.
- Benchmark runs are executed with a fixed seed.
- Artifact paths are writable for all report outputs.

## Limitations

- Operator-level metrics are derived from stage and chunk timing, not kernel profilers.
- Energy estimates are approximate when hardware counters are unavailable.
- Cache/memory hierarchy effects are inferred from trends rather than hardware PMU counters.
