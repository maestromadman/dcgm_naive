import pandas as pd
import matplotlib.pyplot as plt
import json

# Load DCGM log
rows = []
with open("results/dcgm_log.csv") as f:
    for line in f:
        line = line.strip()
        if line.startswith("#") or line.startswith("ID") or not line.startswith("GPU"):
            continue
        parts = line.split()
        if len(parts) == 7:
            try:
                rows.append({
                    'gpu_util': float(parts[2]),
                    'mem_util': float(parts[3]),
                    'fb_mem':   float(parts[4]),
                    'power':    float(parts[5]),
                    'sm_clock': float(parts[6]),
                })
            except:
                continue

df = pd.DataFrame(rows)
df['time_s'] = df.index * 0.2

# Benchmark results
results = {
    'FAISS CPU': 49.62,
    'cuVS IVF-Flat': 0.45,
    'cuVS CAGRA': 0.17
}

# ── Plot 1: DCGM Time Series ──────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle('DCGM GPU Metrics — NVIDIA L4 During ANN Benchmark', fontsize=13, fontweight='bold')

axes[0].plot(df['time_s'], df['gpu_util'],  color='steelblue',  label='GPU Util %')
axes[0].plot(df['time_s'], df['mem_util'],  color='darkorange', label='Mem BW Util %', alpha=0.8)
axes[0].set_ylabel('Utilization %')
axes[0].set_ylim(0, 110)
axes[0].legend(loc='upper right')

axes[1].plot(df['time_s'], df['power'], color='crimson', label='Power (W)')
axes[1].axhline(y=72, color='black', linestyle='--', linewidth=1, label='L4 TDP (72W)')
axes[1].set_ylabel('Power (W)')
axes[1].legend(loc='upper right')

axes[2].plot(df['time_s'], df['sm_clock'], color='seagreen', label='SM Clock (MHz)')
axes[2].set_ylabel('SM Clock (MHz)')
axes[2].set_xlabel('Time (seconds)')
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.savefig('results/dcgm_timeseries.png', dpi=150)
print("Saved dcgm_timeseries.png")

# ── Plot 2: Search Time Comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#d9534f', '#5bc0de', '#5cb85c']
bars = ax.bar(results.keys(), results.values(), color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Search Time (seconds)')
ax.set_title('ANN Search Time: FAISS CPU vs cuVS on NVIDIA L4\n(500k vectors, 128 dimensions, top-10 results)')
ax.set_yscale('log')

for bar, val in zip(bars, results.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
            f'{val}s', ha='center', va='bottom', fontweight='bold')

speedup_ivf   = round(49.62 / 0.45)
speedup_cagra = round(49.62 / 0.17)
ax.text(0.98, 0.95, f'cuVS IVF-Flat: {speedup_ivf}x faster than CPU\ncuVS CAGRA: {speedup_cagra}x faster than CPU',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('results/search_comparison.png', dpi=150)
print("Saved search_comparison.png")

print("\nDone. Check results/ folder for plots.")
