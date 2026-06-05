"""Improved Simulated Annealing (ISA) heuristic."""

import random
import time

import numpy as np

from aeos.greedy import build_initial_solution
from aeos.objective import deterministic_profit, monte_carlo_profit
from aeos.perturbation import perturb_and_reinsert


#  IMPROVED SIMULATED ANNEALING (ISA)

# "The outer loop controls temperature cooling.
#  The inner loop (Markov chain) explores at a fixed temperature.
#  The Metropolis criterion (Eq.18) allows accepting worse solutions
#  with probability e^(Δf/T) — this lets us escape local optima.
#  Temperature decreases by αT=0.95 each outer iteration (cooling schedule).
#  Chain length increases by αL=1.05 — we explore more at lower temperatures."

def run_isa(data,
            T0=100.0,        # Initial temperature
            alpha_T=0.95,    # Cooling rate (T ← T × αT)
            alpha_L=1.05,    # Chain growth rate
            gamma=0.10,      # Disturbance rate
            n_Fm=80,         # Max outer iterations
            zeta_m=0.05,     # Improvement threshold
            n_FTm=20,        # Max rejections before inner loop ends
            n_scenarios=20,  # Monte Carlo scenarios per evaluation
            verbose=True):
    """
    Run the Improved Simulated Annealing heuristic.
    Returns (best_schedule, history, f_mc) where history tracks profit over time.
    """

    print("\nRunning ISA...")
    t_start = time.time()

    # Build initial solution (Stage 1 of ISA)
    schedule = build_initial_solution(data)

    # Initial profit evaluation
    f = deterministic_profit(schedule, data)
    f_best = f
    S_best = schedule.copy()

    # ISA parameters (Algorithm 1)
    T       = T0
    L_T     = max(10, data['n_targets'] // 4)   # Initial chain length
    n_F     = 0     # Total rejected solutions counter
    n_iter  = 0     # Total iterations counter

    history = {
        'profit':       [f],
        'best_profit':  [f_best],
        'temperature':  [T],
        'n_assigned':   [schedule.n_assigned()],
        'iteration':    [0],
    }

    if verbose:
        print(f"  {'Iter':>5} | {'Temp':>8} | {'f':>8} | "
              f"{'f_best':>8} | {'Assigned':>8} | {'Accept':>6}")
        print(f"  {'-'*55}")

    for outer in range(n_Fm):
        n_FT    = 0     # Rejections in this inner loop
        n_iter_T = 0    # Iterations in this inner loop

        # Inner loop (Markov chain at temperature T)
        while n_FT < n_FTm and n_iter_T < L_T:

            # Step 1: Perturb current solution
            S_new = perturb_and_reinsert(schedule, data, gamma)

            # Step 2: Evaluate new solution (fast deterministic proxy)
            f_new = deterministic_profit(S_new, data)

            # Step 3: Metropolis acceptance criterion (Eq.18)
            delta_f = f_new - f

            if delta_f >= 0:
                # Always accept improvements
                accept = True
                Pa = 1.0
            else:
                # Accept worse solution with probability e^(Δf/T)
                Pa = np.exp(delta_f / T)
                accept = random.random() < Pa

            if accept:
                schedule = S_new
                f = f_new

                # Update improvement ratio ζ (Eq.20)
                zeta = delta_f / max(abs(f), 1e-6)
                if zeta >= zeta_m:
                    n_FT = 0    # Good improvement — reset counter
                else:
                    n_FT += 1

                # Update global best
                if f > f_best:
                    f_best = f
                    S_best = schedule.copy()
            else:
                n_FT += 1

            n_iter_T += 1
            n_iter   += 1

        # End of inner loop
        n_F += n_FT

        # Log progress
        history['profit'].append(f)
        history['best_profit'].append(f_best)
        history['temperature'].append(T)
        history['n_assigned'].append(schedule.n_assigned())
        history['iteration'].append(n_iter)

        if verbose and outer % 10 == 0:
            print(f"  {outer:>5} | {T:>8.2f} | {f:>8.2f} | "
                  f"{f_best:>8.2f} | {schedule.n_assigned():>8} | "
                  f"{Pa:>6.3f}")

        # Cooling schedule (Eq. line 19 in Algorithm 1)
        T   = T * alpha_T        # Temperature decreases
        L_T = int(L_T * alpha_L) # Chain length increases

    elapsed = time.time() - t_start

    # Final evaluation with full Monte Carlo
    f_mc = monte_carlo_profit(S_best, data, n_scenarios=50, alpha=0.10)

    print(f"\n  ISA complete in {elapsed:.1f}s")
    print(f"  Targets assigned: {S_best.n_assigned()} / {data['n_targets']}")
    print(f"  Deterministic profit (best): {f_best:.2f}")
    print(f"  Monte Carlo 90% confidence profit: {f_mc:.2f}")

    return S_best, history, f_mc
