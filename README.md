# cuVS ANN Search Analysis – NVIDIA DCGM

## Overview

This project uses NVIDIA DCGM (Data Center GPU Manager) to monitor and analyze real-time GPU behavior while running approximate nearest neighbor (ANN) vector search on an NVIDIA L4 GPU. DCGM samples five hardware metrics at 100ms intervals throughout the benchmark, revealing hardware bottlenecks that timing alone cannot expose.

The algorithm under study is cuVS CAGRA (Compressed, Accurate, GPU-Resident ANN), NVIDIA's graph-based ANN algorithm. Three configurations are benchmarked sequentially on 500,000 MS MARCO passage embeddings (768 dimensions, encoded with BAAI/bge-base-en-v1.5) to study how graph sparsity affects GPU hardware behavior.

---

## Setup

| Component | Details |
|---|---|
| GPU | NVIDIA L4 |
| Instance | GCP VM |
| DCGM Version | 3.3.9 |
| cuVS Version | 25.02 |
| Algorithm | cuVS CAGRA |
| Dataset | 500,000 MS MARCO passages, 768 dimensions |
| Encoder | BAAI/bge-base-en-v1.5 (L2-normalized) |
| Queries | 1,000 vectors, top-10 results |
| DCGM Sampling | 100ms intervals |

---

## DCGM Metrics Monitored

| DCGM Field | What It Measures |
|---|---|
| GPUTL | GPU compute utilization % |
| MCUTL | Memory bandwidth utilization % |
| FBUSD | GPU framebuffer memory used (MB) |
| POWER | Power draw (W) |
| SMCLK | SM clock frequency (MHz) |

---

## Benchmark Phases

Three CAGRA configurations are run sequentially with 3-second pauses between each, creating visible boundaries in the DCGM trace.

| Phase | graph_degree | intermediate_graph_degree | Build Time | Search Time (mean of 5) |
|---|---|---|---|---|
| Phase 1 (baseline) | 64 | 128 | 8.20s | 37.0ms |
| Phase 2 (optimized) | 32 | 64 | 4.42s | 17.3ms |
| Phase 3 (aggressive) | 16 | 32 | 3.45s | 8.3ms |

**Phase 1 to Phase 2:** build time reduced 46%, search time reduced 53%.
**Phase 2 to Phase 3:** build time reduced 22%, search time reduced 52%.

---

## DCGM Findings

### Finding 1: CAGRA build is compute-bound and thermally constrained

During all three build phases, DCGM shows GPU compute utilization leading memory bandwidth utilization by more than 15 percentage points (Phase 1: 49.5% GPU util vs 9.3% mem BW). The build is compute-bound, not memory-bound. Power exceeds the L4's 72W TDP during each build (Phase 1 peak: 74W), and the SM clock drops from its 2040MHz peak to as low as 1095MHz during Phase 1. DCGM identifies this as sustained power throttling: the GPU hits its thermal ceiling and automatically reduces clock speed to stay within the power envelope.

### Finding 2: Reducing graph_degree cuts build duration but does not eliminate throttling

Going from graph_degree=64 to graph_degree=32 cuts build time by 46%. However, DCGM shows that the Phase 2 hardware profile is nearly identical to Phase 1: power still exceeds TDP (74.2W peak), the clock still drops, and the throttle percentage actually increases (30.5% of Phase 1 samples vs 42.2% of Phase 2 samples). The optimization reduces how long the GPU spends at the thermal ceiling, not the ceiling itself. The correct interpretation is that graph_degree=32 does less total work, not that it avoids the bottleneck.

### Finding 3: Diminishing returns beyond graph_degree=32

Going from graph_degree=32 to graph_degree=16 again halves the number of edges per node, but build time falls only 22% (vs 46% for the previous step). The DCGM trace confirms why: the Phase 3 build shows the same power envelope and clock throttle behavior as Phase 2. There is a fixed per-vertex construction cost that does not scale with graph degree, and the GPU hits the same thermal wall regardless of sparsity. graph_degree=32 is the practical sweet spot: further sparsification yields diminishing build improvements while the hardware bottleneck remains unchanged.

### Finding 4: Search phases are too short for meaningful DCGM signal

Search latencies across all three phases (8-37ms) are shorter than or comparable to the 100ms DCGM sampling interval. DCGM cannot meaningfully characterize sub-100ms operations. The search performance improvements (up to 4.5x faster from Phase 1 to Phase 3) are measured via high-resolution Python timers with GPU stream synchronization, not from DCGM. This is a real constraint of polling-based GPU observability tools: they are suited for sustained workloads (builds, training) rather than low-latency inference.

---

## Key Takeaways

| Observation | Implication |
|---|---|
| All builds exceed L4 TDP (72W), triggering clock throttle | CAGRA index construction should be done offline, not on live serving infrastructure |
| Throttle pct increases as graph_degree decreases | Sparser graphs hit the power ceiling just as hard, just faster |
| Build time improvement drops from 46% to 22% across steps | graph_degree=32 is the practical optimization limit on this hardware |
| Search latency halves at each step but is invisible to DCGM | DCGM is the right tool for build profiling; use fine-grained timers for search |

---

## Output Files

| File | Description |
|---|---|
| `results/benchmark_2_results.json` | Build and search times for all three phases |
| `results/dcgm_trace.png` | 4-panel DCGM time series (100ms sampling) with phase annotations |
| `results/dcgm_trace_rolled.png` | Same trace with 500ms rolling average applied for readability |
| `results/build_comparison.png` | Phase-by-phase bar chart of build time, search time, and speedup |
