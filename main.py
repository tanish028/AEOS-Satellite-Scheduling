"""Entry point: run greedy baseline and ISA on a synthetic problem instance."""

import random

import numpy as np

from aeos import (
    generate_problem,
    plot_results,
    plot_temperature,
    run_greedy,
    run_isa,
)
from aeos.config import DEFAULT_SEED


def main():
    np.random.seed(DEFAULT_SEED)
    random.seed(DEFAULT_SEED)

    print("=" * 60)
    print("AEOS Scheduling")
    print("DA 243 — Introduction to Optimization")
    print("=" * 60)

    # Generate problem instance
    data = generate_problem(n_targets=80, n_satellites=2, horizon=1440)

    # Run greedy baseline
    print("\nRunning Greedy Baseline...")
    schedule_greedy, f_greedy_mc = run_greedy(data)

    # Run ISA
    schedule_isa, history, f_isa_mc = run_isa(
        data,
        T0=100.0,
        alpha_T=0.95,
        alpha_L=1.05,
        gamma=0.10,
        n_Fm=80,
        n_scenarios=20,
        verbose=True,
    )

    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Greedy : {schedule_greedy.n_assigned():3d} targets | "
          f"MC 90%-confidence profit = {f_greedy_mc:.2f}")
    print(f"  ISA    : {schedule_isa.n_assigned():3d} targets | "
          f"MC 90%-confidence profit = {f_isa_mc:.2f}")
    improvement = ((f_isa_mc - f_greedy_mc) / max(f_greedy_mc, 1)) * 100
    print(f"  ISA improvement over Greedy: +{improvement:.1f}%")

    # Plots
    print("\nGenerating plots...")
    plot_results(history, schedule_isa, schedule_greedy,
                 f_isa_mc, f_greedy_mc, data)
    plot_temperature(history)


if __name__ == "__main__":
    main()
