"""
analysis_2.py — DCGM trace analysis for benchmark_2.py results.

Reads:
    results/dcgm_log.csv           — raw DCGM metrics at 200ms intervals
    results/benchmark_2_results.json — timing + phase wall-clock timestamps

Outputs:
    results/dcgm_trace.png         — 4-panel DCGM time series with phase annotations
    results/build_comparison.png   — baseline vs optimized: build, search, recall

Bottleneck detection logic:
    Power throttle  — SM clock drops >10% below max while GPU util is high
    Memory-bound    — mem BW util leads GPU compute util by >15 pp
    Compute-bound   — GPU compute util leads mem BW util by >15 pp
"""

import json
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DCGM_CSV    = "results/dcgm_log.csv"
RESULTS_JSON = "results/benchmark_2_results.json"
OUT_DIR     = "results"

# ── 1. Load DCGM log ──────────────────────────────────────────────────────────
if not os.path.exists(DCGM_CSV):
    sys.exit(f"ERROR: {DCGM_CSV} not found. Run DCGM logging before benchmark_2.py.")

rows = []
with open(DCGM_CSV) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("ID"):
            continue
        if not line.startswith("GPU"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            rows.append({
                "gpu_util": float(parts[2]),
                "mem_util": float(parts[3]),
                "fb_mem":   float(parts[4]),
                "power":    float(parts[5]),
                "sm_clock": float(parts[6]),
            })
        except ValueError:
            continue

if not rows:
    sys.exit(f"ERROR: no valid data rows found in {DCGM_CSV}.")

df = pd.DataFrame(rows)
df["time_s"] = df.index * 0.2
print(f"Loaded {len(df)} DCGM samples  ({df['time_s'].max():.1f}s total trace)")

# ── 2. Load benchmark results ─────────────────────────────────────────────────
if not os.path.exists(RESULTS_JSON):
    sys.exit(f"ERROR: {RESULTS_JSON} not found. Run benchmark_2.py first.")

with open(RESULTS_JSON) as f:
    res = json.load(f)

phases    = res["phases"]
baseline  = phases["baseline"]
optimized = phases["optimized"]

# ── 3. Align DCGM trace to benchmark phases ───────────────────────────────────
# Strategy: find the first sample where GPU util exceeds threshold — this is
# when the CAGRA baseline build begins. Then use wall-clock durations from the
# JSON to place the remaining phase boundaries.

IDLE_THRESHOLD = 5.0  # % GPU util considered "idle"

first_active = df[df["gpu_util"] > IDLE_THRESHOLD]["time_s"].iloc[0] if (df["gpu_util"] > IDLE_THRESHOLD).any() else None

if first_active is None:
    print("WARNING: GPU never exceeded idle threshold in DCGM log. Phase annotations disabled.")
    p1_build_start  = None
    p1_build_end    = None
    p1_search_start = None
    p1_search_end   = None
    p2_start        = None
    p2_build_end    = None
    p2_end          = None
else:
    # Phase 1 baseline: build + search
    p1_build_start  = first_active
    p1_build_end    = p1_build_start  + baseline["build_s"]
    p1_search_start = p1_build_end
    # search mean × N_RUNS + warmup (approx)
    p1_search_end   = p1_search_start + baseline["search_mean_s"] * (5 + 1)

    # 3-second gap
    p2_start        = p1_search_end + 3.0
    p2_build_end    = p2_start      + optimized["build_s"]
    p2_search_end   = p2_build_end  + optimized["search_mean_s"] * (5 + 1)
    p2_end          = p2_search_end

    print(f"\nPhase alignment (relative to DCGM log start):")
    print(f"  Phase 1 build    {p1_build_start:.1f}s – {p1_build_end:.1f}s")
    print(f"  Phase 1 search   {p1_search_start:.1f}s – {p1_search_end:.1f}s")
    print(f"  Phase 2 build    {p2_start:.1f}s – {p2_build_end:.1f}s")
    print(f"  Phase 2 search   {p2_build_end:.1f}s – {p2_end:.1f}s")

# ── 4. Bottleneck detection ───────────────────────────────────────────────────
sm_max = df["sm_clock"].max()
sm_min = df["sm_clock"].min()

# Throttle: clock drops >10% below peak while GPU is busy
throttle_mask = (df["gpu_util"] > 80) & (df["sm_clock"] < sm_max * 0.90)
throttle_pct  = throttle_mask.mean() * 100

# Phase-specific analysis
def phase_stats(t_start, t_end):
    if t_start is None:
        return {}
    mask = (df["time_s"] >= t_start) & (df["time_s"] < t_end)
    sub  = df[mask]
    if sub.empty:
        return {}
    return {
        "gpu_util_mean":  sub["gpu_util"].mean(),
        "mem_util_mean":  sub["mem_util"].mean(),
        "power_mean":     sub["power"].mean(),
        "power_max":      sub["power"].max(),
        "sm_clock_mean":  sub["sm_clock"].mean(),
        "sm_clock_min":   sub["sm_clock"].min(),
        "throttle_pct":   ((sub["gpu_util"] > 80) & (sub["sm_clock"] < sm_max * 0.90)).mean() * 100,
    }

p1b_stats = phase_stats(p1_build_start, p1_build_end)
p2b_stats = phase_stats(p2_start,       p2_build_end)

print("\n── Bottleneck Analysis ──────────────────────────────────────────────")
print(f"  SM clock range in trace: {sm_min:.0f} – {sm_max:.0f} MHz")
print(f"  Samples with power throttle (GPU util >80%, clock <90% peak): {throttle_pct:.1f}%")

if p1b_stats:
    print(f"\n  Phase 1 build (graph_degree=64):")
    print(f"    GPU util mean : {p1b_stats['gpu_util_mean']:.1f}%")
    print(f"    Mem BW mean   : {p1b_stats['mem_util_mean']:.1f}%")
    print(f"    Power mean/max: {p1b_stats['power_mean']:.1f}W / {p1b_stats['power_max']:.1f}W")
    print(f"    SM clock mean : {p1b_stats['sm_clock_mean']:.0f} MHz  (min {p1b_stats['sm_clock_min']:.0f} MHz)")
    print(f"    Throttle pct  : {p1b_stats['throttle_pct']:.1f}%")
    if p1b_stats["throttle_pct"] > 10:
        print("    >> FINDING: sustained power throttle — build is thermally constrained")
    if p1b_stats["mem_util_mean"] > p1b_stats["gpu_util_mean"] + 15:
        print("    >> FINDING: memory-bandwidth bound (mem BW > GPU util by >15pp)")
    elif p1b_stats["gpu_util_mean"] > p1b_stats["mem_util_mean"] + 15:
        print("    >> FINDING: compute-bound (GPU util > mem BW by >15pp)")

if p2b_stats:
    print(f"\n  Phase 2 build (graph_degree=32):")
    print(f"    GPU util mean : {p2b_stats['gpu_util_mean']:.1f}%")
    print(f"    Power mean/max: {p2b_stats['power_mean']:.1f}W / {p2b_stats['power_max']:.1f}W")
    print(f"    SM clock mean : {p2b_stats['sm_clock_mean']:.0f} MHz  (min {p2b_stats['sm_clock_min']:.0f} MHz)")
    print(f"    Throttle pct  : {p2b_stats['throttle_pct']:.1f}%")

if p1b_stats and p2b_stats:
    power_reduction = (p1b_stats["power_max"] - p2b_stats["power_max"]) / p1b_stats["power_max"] * 100
    throttle_reduction = p1b_stats["throttle_pct"] - p2b_stats["throttle_pct"]
    print(f"\n  Optimization effect:")
    print(f"    Peak power reduction : {power_reduction:.1f}%")
    print(f"    Throttle reduction   : {throttle_reduction:.1f}pp")
    print(f"    Build time reduction : {(baseline['build_s'] - optimized['build_s']) / baseline['build_s'] * 100:.1f}%")
    print(f"    Recall drop          : {(baseline['recall'] - optimized['recall']):.3f} ({(baseline['recall'] - optimized['recall']) / baseline['recall'] * 100:.1f}%)")

# ── 5. Plot 1: DCGM 4-panel time series ───────────────────────────────────────
L4_TDP = 72.0

fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
fig.suptitle(
    f"DCGM GPU Metrics — NVIDIA L4  |  cuVS CAGRA on MS MARCO 500K×{res['dim']}\n"
    f"Phase 1: graph_degree=64 (baseline)   Phase 2: graph_degree=32 (optimized)",
    fontsize=12, fontweight="bold",
)

PHASE_COLORS = {
    "p1_build":  "#d9534f",
    "p1_search": "#f0ad4e",
    "p2_build":  "#5bc0de",
    "p2_search": "#5cb85c",
}

def shade_phase(ax, t0, t1, color, label=None):
    if t0 is None or t1 is None:
        return
    ax.axvspan(t0, t1, alpha=0.12, color=color, label=label)

# GPU util + Mem BW
axes[0].plot(df["time_s"], df["gpu_util"],  color="steelblue",  label="GPU Compute %", linewidth=1.2)
axes[0].plot(df["time_s"], df["mem_util"],  color="darkorange", label="Mem BW %", alpha=0.8, linewidth=1.2)
axes[0].set_ylabel("Utilization %")
axes[0].set_ylim(0, 110)
axes[0].legend(loc="upper right", fontsize=9)

# Power
axes[1].plot(df["time_s"], df["power"], color="crimson", label="Power (W)", linewidth=1.2)
axes[1].axhline(y=L4_TDP, color="black", linestyle="--", linewidth=1, label=f"L4 TDP ({L4_TDP}W)")
axes[1].set_ylabel("Power (W)")
axes[1].legend(loc="upper right", fontsize=9)

# SM Clock
axes[2].plot(df["time_s"], df["sm_clock"], color="seagreen", label="SM Clock (MHz)", linewidth=1.2)
axes[2].axhline(y=sm_max, color="gray", linestyle=":", linewidth=0.8, label=f"Peak {sm_max:.0f} MHz")
axes[2].set_ylabel("SM Clock (MHz)")
axes[2].legend(loc="upper right", fontsize=9)

# Framebuffer memory
axes[3].plot(df["time_s"], df["fb_mem"], color="purple", label="GPU Mem Used (MB)", linewidth=1.2)
axes[3].set_ylabel("VRAM (MB)")
axes[3].set_xlabel("Time (seconds)")
axes[3].legend(loc="upper right", fontsize=9)

# Phase shading on all panels
for ax in axes:
    shade_phase(ax, p1_build_start,  p1_build_end,   PHASE_COLORS["p1_build"],  "P1 build")
    shade_phase(ax, p1_search_start, p1_search_end,  PHASE_COLORS["p1_search"], "P1 search")
    shade_phase(ax, p2_start,        p2_build_end,   PHASE_COLORS["p2_build"],  "P2 build")
    shade_phase(ax, p2_build_end,    p2_end,         PHASE_COLORS["p2_search"], "P2 search")

legend_patches = [
    mpatches.Patch(color=PHASE_COLORS["p1_build"],  alpha=0.4, label="Phase 1 — build  (gd=64)"),
    mpatches.Patch(color=PHASE_COLORS["p1_search"], alpha=0.4, label="Phase 1 — search (gd=64)"),
    mpatches.Patch(color=PHASE_COLORS["p2_build"],  alpha=0.4, label="Phase 2 — build  (gd=32)"),
    mpatches.Patch(color=PHASE_COLORS["p2_search"], alpha=0.4, label="Phase 2 — search (gd=32)"),
]
axes[0].legend(
    handles=axes[0].get_legend().legend_handles + legend_patches,
    loc="upper right", fontsize=8, ncol=2,
)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "dcgm_trace.png")
plt.savefig(out1, dpi=150)
print(f"\nSaved {out1}")

# ── 6. Plot 2: Before / After comparison ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle(
    f"cuVS CAGRA: Baseline vs Optimized — MS MARCO 500K×{res['dim']}",
    fontsize=13, fontweight="bold",
)

labels  = ["Baseline\n(graph_degree=64)", "Optimized\n(graph_degree=32)"]
colors  = ["#d9534f", "#5cb85c"]

def bar_pair(ax, values, ylabel, title, fmt=".4f", note=None):
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6, width=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.03,
                f"{val:{fmt}}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    if note:
        ax.text(0.5, 0.97, note, transform=ax.transAxes,
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

build_improvement  = (baseline["build_s"]   - optimized["build_s"])   / baseline["build_s"]   * 100
search_improvement = (baseline["search_mean_s"] - optimized["search_mean_s"]) / baseline["search_mean_s"] * 100
recall_drop        = (baseline["recall"]     - optimized["recall"])

bar_pair(axes[0], [baseline["build_s"],       optimized["build_s"]],
         "Build time (s)",   "Index Build Time",
         note=f"{build_improvement:+.1f}% faster")

bar_pair(axes[1], [baseline["search_mean_s"], optimized["search_mean_s"]],
         "Search time (s)", f"Search Time (mean of {N_RUNS} runs)",
         note=f"{search_improvement:+.1f}% faster" if abs(search_improvement) > 1 else "comparable")

bar_pair(axes[2], [baseline["recall"],        optimized["recall"]],
         "Recall@10",       "Recall@10 vs Brute-Force GT",
         fmt=".3f",
         note=f"Δ = {-recall_drop:.3f}  ({recall_drop / baseline['recall'] * 100:.1f}% drop)")

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "build_comparison.png")
plt.savefig(out2, dpi=150)
print(f"Saved {out2}")

print("\nDone. Check results/ for plots and benchmark_2_results.json.")
