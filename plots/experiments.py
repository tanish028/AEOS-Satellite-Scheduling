"""Experiment runners for comparison plot generation."""

import random

import numpy as np

from aeos import generate_problem, run_greedy, run_isa
from plots.config import N_FM, N_SCENARIOS_FINAL, N_SCENARIOS_ISA, SEED


def run_one_experiment(n_targets, gamma=0.10, E_cap=80000, M_cap=7500, seed=SEED):
    """
    One complete experiment:
        generate_problem → run_greedy → run_isa

    Parameters

    n_targets : number of ground targets
    gamma     : ISA disturbance rate γ  (Eq.19)
    E_cap     : energy capacity per orbit in Joules  (Eₖ)
    M_cap     : memory capacity per orbit in MB      (Mₖ)
    seed      : random seed for reproducibility
    Returns
    f_greedy     : float    — greedy 90%-confidence profit
    f_isa        : float    — ISA   90%-confidence profit
    data         : dict     — full problem instance
    schedule_isa : Schedule — best schedule found by ISA
    """
    random.seed(seed)
    np.random.seed(seed)

    data = generate_problem(
        n_targets=n_targets,
        n_satellites=2,
        horizon=1440
    )

    for orb in data['orbits']:
        orb['E_cap'] = E_cap
        orb['M_cap'] = M_cap

    # Run greedy
    schedule_greedy, f_greedy = run_greedy(data)

    # Run ISA
    schedule_isa, history, f_isa = run_isa(
        data,
        T0=100.0,
        alpha_T=0.95,
        alpha_L=1.05,
        gamma=gamma,
        n_Fm=N_FM,
        n_scenarios=N_SCENARIOS_ISA,
        verbose=False,
    )

    return f_greedy, f_isa, data, schedule_isa


def collect_scenario_profits(schedule_isa, data, n_scenarios=N_SCENARIOS_FINAL):
    """
    Collect individual per-scenario profits from the final ISA schedule.
    In each scenario: λˡᵢₖ ~ Bernoulli(pᵢₖ) for every scheduled target.
    These raw values are what we histogram in Plot 3.

    This implements the sample approximation from Section III
    (Eq.15-16), giving us the distribution whose 10th percentile is f.
    """
    profits = []
    for _ in range(n_scenarios):
        total = 0.0
        for k, seq in schedule_isa.assignment.items():
            for (i, tp) in seq:
                p = data['pik'].get((i, k), 0)
                if random.random() < p:          # we will decide on the basis of coin flip cloud or clear
                    total += data['targets'][i]['profit']
        profits.append(total)
    return sorted(profits)
