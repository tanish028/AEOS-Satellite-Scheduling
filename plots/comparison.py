"""Comparison plot generators for ISA vs Greedy analysis."""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from plots.config import (
    BLUE,
    GRAY,
    GREEN,
    N_SCENARIOS_FINAL,
    RED,
    SAVE_DIR,
)
from plots.experiments import collect_scenario_profits, run_one_experiment


def plot_isa_vs_greedy():
    print("\n" + "=" * 55)
    print("Plot 1 — ISA vs Greedy  (n = 50, 75, 100)")
    print("=" * 55)

    n_list    = [50, 75, 100]
    isa_vals  = []
    grdy_vals = []

    for n in n_list:
        print(f"  n={n}: ", end="", flush=True)
        t0 = time.time()
        f_greedy, f_isa, _, _ = run_one_experiment(n_targets=n)
        print(f"{time.time()-t0:.1f}s  |  greedy={f_greedy:.2f}  "
              f"ISA={f_isa:.2f}")
        grdy_vals.append(round(f_greedy, 1))
        isa_vals.append(round(f_isa, 1))

    x     = np.arange(len(n_list))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))

    bars_isa  = ax.bar(x - width/2, isa_vals,  width,
                       color=BLUE, label="ISA (proposed)",  zorder=3)
    bars_grdy = ax.bar(x + width/2, grdy_vals, width,
                       color=GRAY, label="Greedy baseline", zorder=3)

    for bar in bars_isa:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(isa_vals) * 0.02,
                str(round(bar.get_height(), 1)),
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=BLUE)
    for bar in bars_grdy:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(isa_vals) * 0.02,
                str(round(bar.get_height(), 1)),
                ha="center", va="bottom",
                fontsize=11, color=GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in n_list], fontsize=12)
    ax.set_ylim(0, max(isa_vals) * 1.35)
    ax.set_ylabel("90%-confidence profit f", fontsize=11)
    ax.set_title("ISA vs Greedy — confidence profit f",
                 fontsize=14, fontweight="bold", pad=14)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "isa_vs_greedy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: isa_vs_greedy.png")
    return isa_vals, grdy_vals, n_list


def plot_gamma_analysis():
    print("\n" + "=" * 55)
    print("Plot 2 — Gamma analysis  (gamma = 0.05, 0.10, 0.20)")
    print("=" * 55)

    n_list     = [50, 75, 100]
    gamma_list = [0.05, 0.10, 0.20]
    results    = {g: [] for g in gamma_list}

    for n in n_list:
        for g in gamma_list:
            print(f"  n={n}, gamma={g}: ", end="", flush=True)
            t0 = time.time()
            _, f_isa, _, _ = run_one_experiment(n_targets=n, gamma=g)
            print(f"{time.time()-t0:.1f}s  |  ISA={f_isa:.2f}")
            results[g].append(round(f_isa, 1))

    x = np.arange(len(n_list))
    fig, ax = plt.subplots(figsize=(7, 5))

    line_styles = {
        0.05: {"color": GRAY,  "lw": 2.0, "ms": 7},
        0.10: {"color": BLUE,  "lw": 2.5, "ms": 8},
        0.20: {"color": RED,   "lw": 2.0, "ms": 7},
    }

    for g in gamma_list:
        s = line_styles[g]
        ax.plot(x, results[g], "o-",
                color=s["color"], linewidth=s["lw"],
                markersize=s["ms"], label=f"γ={g}", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in n_list], fontsize=12)
    all_vals = [v for vals in results.values() for v in vals]
    ax.set_ylim(0, max(all_vals) * 1.35)
    ax.set_ylabel("90%-confidence profit f", fontsize=11)
    ax.set_title("Effect of disturbance rate γ on profit",
                 fontsize=14, fontweight="bold", pad=14)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "gamma_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: gamma_analysis.png")
    return results, n_list


def plot_mc_distribution():
    print("\n" + "=" * 55)
    print("Plot 3 — Monte Carlo distribution  (n=100)")
    print("=" * 55)

    print("  Running ISA at n=100...", end=" ", flush=True)
    t0 = time.time()
    _, f_isa, data, schedule_isa = run_one_experiment(n_targets=100)
    print(f"{time.time()-t0:.1f}s  |  ISA={f_isa:.2f}")

    print(f"  Collecting {N_SCENARIOS_FINAL} scenario profits...",
          end=" ", flush=True)
    scenario_profits = collect_scenario_profits(
        schedule_isa, data, n_scenarios=N_SCENARIOS_FINAL
    )
    print("done")

    # f = 10th percentile — as it was choosen in the paper
    f_val = float(np.percentile(scenario_profits, 10))
    print(f"  f (10th percentile) = {f_val:.2f}")
    print(f"  Range: {min(scenario_profits):.1f} – {max(scenario_profits):.1f}")

    fig, ax = plt.subplots(figsize=(7, 5))

    lo   = int(min(scenario_profits) // 50) * 50
    hi   = int(max(scenario_profits) // 50) * 50 + 50
    bins = range(lo, hi + 50, 50)

    ax.hist(scenario_profits, bins=bins,
            color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)

    ax.axvline(f_val, color=RED, linewidth=2.2, linestyle="--",
               label=f"f = {f_val:.1f}  (10th percentile)", zorder=5)

    ax.set_xlabel("Profit per scenario", fontsize=11)
    ax.set_ylabel("Number of scenarios",  fontsize=11)
    ax.set_title(
        f"Monte Carlo profit distribution\n(|W| = {N_SCENARIOS_FINAL} scenarios)",
        fontsize=14, fontweight="bold", pad=14)
    ax.legend(frameon=False, fontsize=11)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "mc_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: mc_distribution.png")
    return scenario_profits, f_val


def plot_resource_analysis():
    print("\n" + "=" * 55)
    print("Plot 4 — Resource analysis  (Ek × Mk combinations)")
    print("=" * 55)

    E_caps   = [40000, 80000, 120000]
    M_caps   = [5000,  7500,  10000]
    E_labels = ["Ek=40kJ", "Ek=80kJ", "Ek=120kJ"]
    M_labels = ["Mk=5000", "Mk=7500", "Mk=10000"]

    results = [[0.0] * len(M_caps) for _ in range(len(E_caps))]

    for ei, E_cap in enumerate(E_caps):
        for mi, M_cap in enumerate(M_caps):
            print(f"  Ek={E_cap//1000}kJ, Mk={M_cap}MB: ",
                  end="", flush=True)
            t0 = time.time()
            _, f_isa, _, _ = run_one_experiment(
                n_targets=100,
                E_cap=E_cap,
                M_cap=M_cap,
            )
            print(f"{time.time()-t0:.1f}s  |  ISA={f_isa:.2f}")
            results[ei][mi] = round(f_isa, 1)

    # profit vs energy and memory
    x      = np.arange(len(M_caps))
    width  = 0.25
    colors = [GRAY, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(7, 5))

    for ei, (E_label, color) in enumerate(zip(E_labels, colors)):
        offset = (ei - 1) * width      # -width, 0, +width centres the group
        ax.bar(x + offset, results[ei], width,
               color=color, label=E_label, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(M_labels, fontsize=12)
    all_vals = [v for row in results for v in row]
    ax.set_ylim(0, max(all_vals) * 1.35)
    ax.set_ylabel("90%-confidence profit f", fontsize=11)
    ax.set_title("Profit vs Energy & Memory constraints",
                 fontsize=14, fontweight="bold", pad=14)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "resource_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: resource_analysis.png")
    return results, E_labels, M_labels
