"""
benchmark_2.py — cuVS CAGRA on MS MARCO embeddings with DCGM profiling.

Loads 500K × 768 embeddings produced by prepare_data.py (MS MARCO passages
encoded with BAAI/bge-base-en-v1.5). Falls back to synthetic data if not found.

Runs two CAGRA configurations back-to-back so DCGM can capture both:
  Phase 1  baseline   graph_degree=64  (default — triggers power throttling)
  Phase 2  optimized  graph_degree=32  (reduces build-phase thermal pressure)

Ground truth is computed via cuVS brute_force (exact inner-product on GPU).

Run with DCGM logging active:
    dcgmi dmon -e 203,204,252,155,100 -d 200 > results/dcgm_log.csv &
    python benchmark_2.py
"""

import os
import sys
import json
import time

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
N_CORPUS  = 500_000
K         = 10
DIM       = 768
DATA_DIR  = "data"
OUT_DIR   = "results"
N_RUNS    = 5        # timed search repetitions per config (after 1 warmup)
SEED      = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
corpus_path  = os.path.join(DATA_DIR, f"embeddings_{DIM}.npy")
queries_path = os.path.join(DATA_DIR, f"queries_{DIM}.npy")

if os.path.exists(corpus_path) and os.path.exists(queries_path):
    print(f"Loading MS MARCO embeddings from {DATA_DIR}/...")
    corpus  = np.load(corpus_path)[:N_CORPUS].astype(np.float32)
    queries = np.load(queries_path).astype(np.float32)
    data_source = "MS MARCO — BAAI/bge-base-en-v1.5 (L2-normalized)"
    print(f"  corpus : {corpus.shape}  ({corpus.nbytes / 1e9:.2f} GB)")
    print(f"  queries: {queries.shape}")
else:
    print(f"[WARNING] {corpus_path} not found — using synthetic random data.")
    print("          Run prepare_data.py first for real MS MARCO embeddings.\n")
    rng     = np.random.default_rng(SEED)
    corpus  = rng.random((N_CORPUS, DIM)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.random((1_000, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    data_source = "synthetic (random, L2-normalized)"

N_QUERIES = len(queries)

# ── Ground truth (FAISS exact inner-product search, CPU) ──────────────────────
print(f"\nComputing brute-force ground truth (FAISS IndexFlatIP, {N_QUERIES} queries)...")
faiss.omp_set_num_threads(os.cpu_count() or 1)
gt_index = faiss.IndexFlatIP(DIM)
gt_index.add(corpus)
t0 = time.perf_counter()
_, I_gt = gt_index.search(queries, K)
print(f"  Done in {time.perf_counter() - t0:.2f}s")

def recall_at_k(I_pred_gpu):
    I_pred = cp.asnumpy(I_pred_gpu)
    hits = sum(len(set(I_gt[i]) & set(I_pred[i])) for i in range(N_QUERIES))
    return hits / (N_QUERIES * K)

# ── GPU setup ─────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    from cuvs.neighbors import cagra
    from cuvs.common.resources import Resources
except ImportError as e:
    sys.exit(f"GPU libraries unavailable: {e}")

res         = Resources()
corpus_gpu  = cp.asarray(corpus)
queries_gpu = cp.asarray(queries)

def sync():
    cp.cuda.Stream.null.synchronize()

def timed_search(fn):
    fn(); sync()                    # warmup
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn(); sync()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))

# ── Results container ─────────────────────────────────────────────────────────
results = {
    "data_source": data_source,
    "n_corpus":    N_CORPUS,
    "n_queries":   N_QUERIES,
    "dim":         DIM,
    "k":           K,
    "phases":      {},
}

# ── Phase 1: CAGRA baseline  (graph_degree=64) ───────────────────────────────
print("\n" + "=" * 60)
print("Phase 1: CAGRA baseline  (graph_degree=64, intermediate=128)")
print("=" * 60)

results["phases"]["baseline_wall_start"] = time.time()

bp1  = cagra.IndexParams(graph_degree=64, intermediate_graph_degree=128,
                          metric="inner_product")
t0   = time.perf_counter()
idx1 = cagra.build(bp1, corpus_gpu, resources=res)
sync()
build1 = time.perf_counter() - t0
print(f"  Build : {build1:.4f}s")

sp1              = cagra.SearchParams()
mean1, std1      = timed_search(lambda: cagra.search(sp1, idx1, queries_gpu, K, resources=res))
_, I1            = cagra.search(sp1, idx1, queries_gpu, K, resources=res); sync()
rec1             = recall_at_k(I1)
print(f"  Search: {mean1:.4f}s ± {std1:.4f}s   recall@{K}={rec1:.3f}")

results["phases"]["baseline_wall_end"] = time.time()
results["phases"]["baseline"] = {
    "graph_degree": 64, "intermediate_graph_degree": 128,
    "build_s": round(build1, 4),
    "search_mean_s": round(mean1, 4), "search_std_s": round(std1, 4),
    "recall": round(rec1, 4),
}

time.sleep(3)

# ── Phase 2: CAGRA optimized  (graph_degree=32) ──────────────────────────────
print("\n" + "=" * 60)
print("Phase 2: CAGRA optimized  (graph_degree=32, intermediate=64)")
print("         Hypothesis: fewer edges → less build-phase thermal pressure")
print("=" * 60)

results["phases"]["optimized_wall_start"] = time.time()

bp2  = cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64,
                          metric="inner_product")
t0   = time.perf_counter()
idx2 = cagra.build(bp2, corpus_gpu, resources=res)
sync()
build2 = time.perf_counter() - t0
print(f"  Build : {build2:.4f}s")

sp2              = cagra.SearchParams()
mean2, std2      = timed_search(lambda: cagra.search(sp2, idx2, queries_gpu, K, resources=res))
_, I2            = cagra.search(sp2, idx2, queries_gpu, K, resources=res); sync()
rec2             = recall_at_k(I2)
print(f"  Search: {mean2:.4f}s ± {std2:.4f}s   recall@{K}={rec2:.3f}")

results["phases"]["optimized_wall_end"] = time.time()
results["phases"]["optimized"] = {
    "graph_degree": 32, "intermediate_graph_degree": 64,
    "build_s": round(build2, 4),
    "search_mean_s": round(mean2, 4), "search_std_s": round(std2, 4),
    "recall": round(rec2, 4),
}

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "benchmark_2_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
b_delta  = (build1  - build2)  / build1  * 100
s_delta  = (mean1   - mean2)   / mean1   * 100
rec_drop = (rec1    - rec2)    / rec1    * 100

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"  {'':25s} {'baseline':>12} {'optimized':>12} {'delta':>10}")
print(f"  {'-' * 61}")
print(f"  {'graph_degree':25s} {'64':>12} {'32':>12}")
print(f"  {'build (s)':25s} {build1:>12.4f} {build2:>12.4f} {-b_delta:>+9.1f}%")
print(f"  {'search mean (s)':25s} {mean1:>12.4f} {mean2:>12.4f} {-s_delta:>+9.1f}%")
print(f"  {'recall@10':25s} {rec1:>12.3f} {rec2:>12.3f} {-rec_drop:>+9.1f}%")
print(f"\nResults written to {out_path}")
print("Run  python analysis_2.py  to generate DCGM plots.")
