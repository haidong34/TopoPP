# TopoPP

TopoPP selects contiguous layer partitions for pipeline-parallel inference under uneven GPU performance. It applies partitions through `VLLM_PP_LAYER_PARTITION` in official vLLM.

- **V1:** use per-GPU GEMM latency as a per-layer cost proxy.
- **V2:** measure actual vLLM stage execution and validate partitions with serving benchmarks.

These are project versions; V2 uses the **vLLM V1 engine**.

## V1: GEMM proxy

V1 estimates stage cost as `layer_count × GEMM_latency` and uses dynamic programming to minimize the largest stage cost. Profiling uses 100 GEMMs per timing sample to reduce short-window measurement issues under contention.

The experiment used **Qwen3-0.6B, 28 layers, TP=1 and PP=4**, with continuous BF16 `8192³` GEMM contention on GPU 2. The partition changed from `[7,7,7,7]` to `[8,8,4,8]`.

| Metric | Default | Selected |
|---|---:|---:|
| Output throughput (tok/s) | 593.03 | 579.32 |
| Median TTFT (ms) | 82.11 | 81.05 |
| Median TPOT (ms) | 13.17 | 13.32 |
| P99 ITL (ms) | 16.49 | 24.72 |

The proxy predicted a **42.86% bottleneck reduction**, but measured throughput decreased by **2.31%** in this single comparison. This motivated measuring actual stage execution in V2.

## V2: vLLM stage profiling

V2 collects CUDA-event timings around local `GPUModelRunner.execute_model` execution. It builds stage-cost curves for prefill, decode and mixed workload buckets, then minimizes the weighted sum of their stage bottlenecks. Weights come from the default calibration run. Search respects measured layer-count ranges and estimated memory budgets.

Costs use **local CUDA time**; host elapsed time is diagnostic only because it can include upstream waits. Calibration preserves CUDA Graph settings, and final A/B runs use the stock vLLM worker.

### Setup

- **Hardware:** one node, 4 × A800-SXM4-80GB, NV8 links.
- **Runtime:** official vLLM 0.10.1.1, PyTorch 2.7.1+cu126, CUDA runtime 12.6.
- **Model:** Qwen3-14B, 40 layers, BF16, TP=1, PP=4, DP=1.
- **Workload:** 1,024 input tokens, 128 output tokens, concurrency 8, 128 requests per A/B run.
- **KV pool:** fixed 16,384 token slots; prefix caching disabled.
- **Calibration:** `[10,10,10,10]`, `[16,4,16,4]`, `[4,16,4,16]`; 99.46% reference-iteration coverage in both conditions.
- **Conditions:** H0 has no contention; H1 runs continuous BF16 `8192³` GEMM on GPU 2.
- **A/B order:** A, B, B, A, A, B; three runs per scheme. A uses `[10,10,10,10]`.

### Results

| Condition | Selected partition | Default throughput | Selected throughput | Change |
|---|---|---:|---:|---:|
| H0 | `[9,11,11,9]` | 269.29 tok/s | 269.91 tok/s | +0.23% |
| H1 | `[11,12,6,11]` | 212.71 tok/s | 221.81 tok/s | **+4.28%** |

| Condition | Metric | Default | Selected | Change |
|---|---|---:|---:|---:|
| H0 | Median TTFT (ms) | 940.10 | 937.48 | −0.28% |
| H0 | Median TPOT (ms) | 22.55 | 22.43 | −0.53% |
| H0 | P99 ITL (ms) | 24.51 | 24.58 | +0.29% |
| H1 | Median TTFT (ms) | 1,239.84 | 1,116.09 | −9.98% |
| H1 | Median TPOT (ms) | 28.13 | 27.54 | −2.10% |
| H1 | P99 ITL (ms) | 30.80 | 29.73 | −3.47% |

Values are medians across three per-run metrics, including per-run P99 values. H1's paired throughput gains were **+4.01%, +4.05% and +4.80%**. H0 is treated as approximately unchanged.

The benefit is limited to the tested workload and contention condition. Three pairs do not establish statistical significance. V1 and V2 use different models, so their results are not a controlled ablation.

## Run

Each version directory contains scripts, dependencies, tests and reported results. V2 also contains the experiment configuration. GPU experiments run on Linux; download model weights separately.

**V1 planner example** — no GPU required:

```bash
cd V1
python scripts/partition.py --profile examples/h1_proxy_ratios.json --num-layers 28
python scripts/verify_partition.py --profile examples/h1_proxy_ratios.json --num-layers 28
```

**V2 experiment** — install the pinned runtime and update the model path in [the configuration](V2/configs/a800_qwen3_14b.json):

```bash
cd V2
python -m pip install -r requirements.txt
python scripts/server_a800_14b.py preflight --run-dir results/a800_14b/preflight_01
python scripts/server_a800_14b.py smoke --run-dir results/a800_14b/smoke_01

# Run H0, then H1. The driver manages the H1 competitor.
for state in H0 H1; do
  contention=()
  if [ "$state" = H1 ]; then contention=(--with-contention); fi
  calibration="results/a800_14b/cal_${state}_01"
  python scripts/server_a800_14b.py calibrate --state "$state" "${contention[@]}" --run-dir "$calibration"
  python scripts/server_a800_14b.py plan --state "$state" --run-dir "$calibration"
  python scripts/server_a800_14b.py ab --state "$state" "${contention[@]}" \
    --partition-file "$calibration/partition.json" --run-dir "results/a800_14b/ab_${state}_01"
done
```

Use fresh directories for GPU runs; `plan` reuses its calibration directory. Selected partitions are saved in `partition.json`, and A/B results in `ab_summary.json`. Use each script's `--help` for options.

**CPU checks** — run separately from the repository root:

```bash
python -m unittest discover -s V1/tests -v
python -m unittest discover -s V2/tests -v
```
