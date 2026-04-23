import os, sys, json, time
import numpy as np
import cupy as cp
from cuvs.neighbors import cagra
from cuvs.common.resources import Resources

N_CORPUS = 500_000
K        = 10
DIM      = 768
DATA_DIR = "data"
OUT_DIR  = "results"
N_RUNS   = 5

os.makedirs(OUT_DIR, exist_ok=True)

corpus  = np.load(os.path.join(DATA_DIR, f"embeddings_{DIM}.npy"))[:N_CORPUS].astype(np.float32)
queries = np.load(os.path.join(DATA_DIR, f"queries_{DIM}.npy")).astype(np.float32)
print(f"corpus: {corpus.shape}  queries: {queries.shape}")

res         = Resources()
corpus_gpu  = cp.asarray(corpus)
queries_gpu = cp.asarray(queries)

def sync():
    cp.cuda.Stream.null.synchronize()

def timed_search(fn):
    fn(); sync()
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn(); sync()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))

results = {"phases": {}}

print("\n" + "="*60)
print("Phase 1: CAGRA baseline  (graph_degree=64, intermediate=128)")
print("="*60)
results["phases"]["baseline_wall_start"] = time.time()
bp1    = cagra.IndexParams(graph_degree=64, intermediate_graph_degree=128, metric="inner_product")
t0     = time.perf_counter()
idx1   = cagra.build(bp1, corpus_gpu, resources=res)
sync()
build1 = time.perf_counter() - t0
print(f"  Build : {build1:.4f}s")
sp1         = cagra.SearchParams()
mean1, std1 = timed_search(lambda: cagra.search(sp1, idx1, queries_gpu, K, resources=res))
print(f"  Search: {mean1:.4f}s ± {std1:.4f}s")
results["phases"]["baseline_wall_end"] = time.time()
results["phases"]["baseline"] = {"graph_degree": 64, "build_s": round(build1,4), "search_mean_s": round(mean1,4), "search_std_s": round(std1,4)}

time.sleep(3)

print("\n" + "="*60)
print("Phase 2: CAGRA optimized  (graph_degree=32, intermediate=64)")
print("="*60)
results["phases"]["optimized_wall_start"] = time.time()
bp2    = cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64, metric="inner_product")
t0     = time.perf_counter()
idx2   = cagra.build(bp2, corpus_gpu, resources=res)
sync()
build2 = time.perf_counter() - t0
print(f"  Build : {build2:.4f}s")
sp2         = cagra.SearchParams()
mean2, std2 = timed_search(lambda: cagra.search(sp2, idx2, queries_gpu, K, resources=res))
print(f"  Search: {mean2:.4f}s ± {std2:.4f}s")
results["phases"]["optimized_wall_end"] = time.time()
results["phases"]["optimized"] = {"graph_degree": 32, "build_s": round(build2,4), "search_mean_s": round(mean2,4), "search_std_s": round(std2,4)}

time.sleep(3)

# ── Phase 3: CAGRA aggressive  (graph_degree=16, intermediate=32) ────────────
print("\n" + "="*60)
print("Phase 3: CAGRA aggressive  (graph_degree=16, intermediate=32)")
print("         Hypothesis: sparser graph → faster build, less throttle")
print("="*60)
results["phases"]["aggressive_wall_start"] = time.time()

bp3    = cagra.IndexParams(graph_degree=16, intermediate_graph_degree=32, metric="inner_product")
t0     = time.perf_counter()
idx3   = cagra.build(bp3, corpus_gpu, resources=res)
sync()
build3 = time.perf_counter() - t0
print(f"  Build : {build3:.4f}s")

sp3         = cagra.SearchParams()
mean3, std3 = timed_search(lambda: cagra.search(sp3, idx3, queries_gpu, K, resources=res))
print(f"  Search: {mean3:.4f}s ± {std3:.4f}s")

results["phases"]["aggressive_wall_end"] = time.time()
results["phases"]["aggressive"] = {
    "graph_degree": 16, "intermediate_graph_degree": 32,
    "build_s": round(build3, 4),
    "search_mean_s": round(mean3, 4), "search_std_s": round(std3, 4),
}

with open(os.path.join(OUT_DIR, "benchmark_2_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  {'':25s} {'gd=64':>10} {'gd=32':>10} {'gd=16':>10}")
print(f"  {'-'*57}")
print(f"  {'build (s)':25s} {build1:>10.4f} {build2:>10.4f} {build3:>10.4f}")
print(f"  {'search mean (s)':25s} {mean1:>10.4f} {mean2:>10.4f} {mean3:>10.4f}")
print(f"  {'search speedup vs P1':25s} {'—':>10} {mean1/mean2:>9.1f}x {mean1/mean3:>9.1f}x")
print("\nDone. Run  python analysis_2.py  to generate DCGM plots.")
