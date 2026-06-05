"""
AEOS Satellite Scheduling under Cloud Coverage Uncertainty.

"We implemented the core ideas from the paper in Python. We simplified
orbital mechanics (using random VTWs instead of real orbits) but kept
the optimization structure identical: binary assignment variables xik,
continuous time-position variables TPik, energy/memory constraints,
Monte Carlo cloud evaluation, and the ISA(Improved Simulated Annealing)
loop with Metropolis acceptance."
"""

from aeos.constraints import check_feasibility
from aeos.data_generation import generate_problem
from aeos.greedy import build_initial_solution, run_greedy
from aeos.isa import run_isa
from aeos.objective import deterministic_profit, monte_carlo_profit
from aeos.physics import compute_obs_start, compute_slew_energy, compute_trans
from aeos.schedule import Schedule
from aeos.selection import compute_conflict, compute_need
from aeos.visualization import plot_results, plot_temperature

__all__ = [
    "Schedule",
    "check_feasibility",
    "compute_conflict",
    "compute_need",
    "compute_obs_start",
    "compute_slew_energy",
    "compute_trans",
    "deterministic_profit",
    "generate_problem",
    "build_initial_solution",
    "monte_carlo_profit",
    "plot_results",
    "plot_temperature",
    "run_greedy",
    "run_isa",
]
