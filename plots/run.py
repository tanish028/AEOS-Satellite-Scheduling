"""
Generate comparison plots for ISA analysis:
       isa_vs_greedy.png      — ISA vs Greedy bar chart (n=50, 75, 100)
       gamma_analysis.png     — Effect of gamma=0.05, 0.10, 0.20
       mc_distribution.png    — Monte Carlo profit histogram
       resource_analysis.png  — Profit vs Energy & Memory constraints

Total runtime is ~5-10 minutes because ISA runs 13 times.
"""

import os
import time

from plots.comparison import (
    plot_gamma_analysis,
    plot_isa_vs_greedy,
    plot_mc_distribution,
    plot_resource_analysis,
)
from plots.config import N_FM, N_SCENARIOS_FINAL, SAVE_DIR, SEED


def main():
    print("=" * 55)
    print("AEOS Result Plot Generator")
    print("DA 243 — Introduction to Optimization")
    print("=" * 55)
    print(f"Output folder : {SAVE_DIR}")
    print(f"ISA iterations: {N_FM}  (N_FM)")
    print(f"MC scenarios  : {N_SCENARIOS_FINAL}  (final histogram)")
    print(f"Random seed   : {SEED}")

    total_t = time.time()

    isa_v, grdy_v, n_l          = plot_isa_vs_greedy()
    gamma_r, _                  = plot_gamma_analysis()
    sc_profits, f_mc            = plot_mc_distribution()
    res_r, e_lbls, m_lbls       = plot_resource_analysis()

    print("\n--- ISA vs Greedy ---")
    for n, isa, grdy in zip(n_l, isa_v, grdy_v):
        imp = (isa - grdy) / max(grdy, 1) * 100
        print(f"  n={n}: Greedy={grdy:.1f}  ISA={isa:.1f}  (+{imp:.1f}%)")

    print("\n--- Gamma (n=100) ---")
    for g in [0.05, 0.10, 0.20]:
        print(f"  gamma={g}: f={gamma_r[g][-1]:.1f}")

    print(f"\n--- MC Distribution (n=100) ---")
    print(f"  {len(sc_profits)} scenarios | "
          f"range {min(sc_profits):.1f}–{max(sc_profits):.1f} | "
          f"f={f_mc:.1f}")

    print("\n--- Resource Analysis (n=100) ---")
    for ei, e in enumerate(e_lbls):
        row = "  ".join(f"{m}:{res_r[ei][mi]:.1f}"
                        for mi, m in enumerate(m_lbls))
        print(f"  {e}: {row}")

    print("\nSaved files:")
    for f in ["isa_vs_greedy.png", "gamma_analysis.png",
              "mc_distribution.png", "resource_analysis.png"]:
        print(f"  {os.path.join(SAVE_DIR, f)}")

    print(f"\nTotal runtime: {time.time() - total_t:.1f}s")


if __name__ == "__main__":
    main()
