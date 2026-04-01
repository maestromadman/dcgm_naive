import numpy as np
import faiss
import time
import json

D = 128
N_TRAIN = 500000
N_QUERY = 10000
K = 10

print("Generating data...")
np.random.seed(42)
data    = np.random.rand(N_TRAIN, D).astype('float32')
queries = np.random.rand(N_QUERY, D).astype('float32')

results = {}

print("\nRunning FAISS CPU (baseline)...")
index_cpu = faiss.IndexFlatL2(D)
index_cpu.add(data)
t0 = time.perf_counter()
_, I_cpu = index_cpu.search(queries, K)
results['faiss_cpu'] = round(time.perf_counter() - t0, 4)
print(f"  Done: {results['faiss_cpu']}s")
time.sleep(3)

print("\nRunning cuVS IVF-Flat...")
import cupy as cp
from cuvs.neighbors import ivf_flat
from cuvs.common.resources import Resources

res = Resources()
data_gpu    = cp.asarray(data)
queries_gpu = cp.asarray(queries)

build_params = ivf_flat.IndexParams(n_lists=1024, metric="sqeuclidean")
t0 = time.perf_counter()
index_ivf = ivf_flat.build(build_params, data_gpu, resources=res)
print(f"  Build: {round(time.perf_counter() - t0, 4)}s")

search_params = ivf_flat.SearchParams(n_probes=50)
t0 = time.perf_counter()
ivf_flat.search(search_params, index_ivf, queries_gpu, K, resources=res)
results['cuvs_ivf_flat'] = round(time.perf_counter() - t0, 4)
print(f"  Search: {results['cuvs_ivf_flat']}s")
time.sleep(3)

print("\nRunning cuVS CAGRA...")
from cuvs.neighbors import cagra

cagra_params = cagra.IndexParams(metric="sqeuclidean")
t0 = time.perf_counter()
index_cagra = cagra.build(cagra_params, data_gpu, resources=res)
print(f"  Build: {round(time.perf_counter() - t0, 4)}s")

cagra_search = cagra.SearchParams()
t0 = time.perf_counter()
cagra.search(cagra_search, index_cagra, queries_gpu, K, resources=res)
results['cuvs_cagra'] = round(time.perf_counter() - t0, 4)
print(f"  Search: {results['cuvs_cagra']}s")
time.sleep(3)

with open("results/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Summary ===")
for k, v in results.items():
    print(f"  {k:20s}: {v}s")
