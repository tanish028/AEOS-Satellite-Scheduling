"""Profit evaluation under cloud coverage uncertainty."""

import random

import numpy as np


#  OBJECTIVE FUNCTION — Monte Carlo Profit Evaluation
#  Instead of maximizing Σ ωᵢ·xᵢₖ (deterministic), we maximize the confidence
#  profit f = the profit achieved in at least 90% of cloud scenarios.
#  We implement this via Monte Carlo: sample n_scenarios random cloud
#  realizations, compute profit in each, and take the 10th percentile.
#  This directly approximates Eq.14: P(profit ≥ f) ≥ 1−α = 0.90.

def monte_carlo_profit(schedule, data, n_scenarios=30, alpha=0.10):
    """
    Evaluate schedule profit under cloud uncertainty.

    Implements the sample approximation from Section III:
    - Generate |W| = 30 cloud scenarios
    - In each scenario, λˡᵢₖ ~ Bernoulli(pᵢₖ)
    - Compute realized profit per scenario
    - Return the (alpha)-quantile as the confidence profit f
    """
    scenario_profits = []

    for _ in range(n_scenarios):
        profit = 0.0
        for k, seq in schedule.assignment.items():
            for (i, tp) in seq:
                p = data['pik'].get((i, k), 0)
                # λˡᵢₖ ~ Bernoulli(pᵢₖ)
                if random.random() < p:
                    profit += data['targets'][i]['profit']
        scenario_profits.append(profit)

    if not scenario_profits:
        return 0.0

    # f = alpha-quantile of scenario profits
    # = profit exceeded in (1-alpha) = 90% of scenarios
    return float(np.percentile(scenario_profits, alpha * 100))


def deterministic_profit(schedule, data):
    """
    Quick deterministic profit: Σ ωᵢ·pᵢₖ·xᵢₖ (expected value).
    Used during SA iterations for speed; Monte Carlo used for final eval.

    """
    profit = 0.0
    for k, seq in schedule.assignment.items():
        for (i, tp) in seq:
            p = data['pik'].get((i, k), 0)
            profit += data['targets'][i]['profit'] * p
    return profit
