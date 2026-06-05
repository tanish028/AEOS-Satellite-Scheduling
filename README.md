# AEOS Satellite Scheduling under Cloud Coverage Uncertainty

Implementation of a Simulated Annealing based heuristic for scheduling Agile Earth Observation Satellites (AEOS) under cloud coverage uncertainty.

**Based on:** Han, Gu, Wu & Wang — *"Simulated Annealing Based Heuristic for Multiple Agile Satellites Scheduling under Cloud Coverage Uncertainty"* — IEEE Transactions on Systems, Man, and Cybernetics: Systems.

**Course:** DA 243 — Introduction to Optimization, 2026  
**Institute:** Mehta Family School of Data Science and Artificial Intelligence, IIT Guwahati

---

## Team

| Name                 | Roll Number |
|----------------------|-------------|
| Tanish Anand         | 240150037   |
| Tumu Sai Seshi Kiran | 240150038   |
| Rathod Ravi Kiran    | 240150029   |

---

## Project Overview

Agile Earth Observation Satellites (AEOS) can rotate in three directions (roll, pitch, yaw), giving them a flexible time window to observe each ground target. This flexibility makes scheduling much more powerful but also NP-hard.

The key challenge: **cloud coverage randomly blocks optical sensors**, so we cannot simply maximize deterministic profit. Instead, we use **Chance Constrained Programming (CCP)** to find schedules that achieve a guaranteed profit level with 90% confidence.

Our implementation includes:
- Synthetic problem generation (targets, orbits, visibility windows, cloud probabilities)
- Greedy baseline scheduler
- Improved Simulated Annealing (ISA) with Metropolis (temperature-based) acceptance
- Monte Carlo profit evaluation under cloud uncertainty
- Result plots comparing ISA against the greedy baseline

---

## Repository Structure

```
AEOS-Satellite-Scheduling/
│
├── aeos/                        # Core scheduling package
│   ├── __init__.py              # Public API exports
│   ├── config.py                # Paths and default settings
│   ├── data_generation.py       # Synthetic problem instance generation
│   ├── physics.py               # Rotation time, slew energy, observation start
│   ├── schedule.py              # Schedule data structure
│   ├── constraints.py           # Feasibility checking (C1–C5)
│   ├── objective.py             # Monte Carlo and deterministic profit
│   ├── selection.py             # Needᵢ and CFᵢₖ scoring rules
│   ├── greedy.py                # Greedy baseline and initial solution
│   ├── perturbation.py          # Delete-and-reinsert perturbation operator
│   ├── isa.py                   # Improved Simulated Annealing loop
│   └── visualization.py         # Convergence and Gantt chart plots
│
├── plots/                       # Comparison plot generation
│   ├── __init__.py
│   ├── config.py                # Plot styling and experiment constants
│   ├── experiments.py           # Single-run experiment helpers
│   └── comparison.py              # Four analysis plot functions
│
├── main.py                      # Primary entry point (ISA + greedy)
├── comparison_plots.py          # Generate all 4 comparison plots
├── Satellite.py                 # Backward-compatible wrapper
├── requirements.txt
├── README.md
│
└── results/                     # Generated plots (created on first run)
    ├── results.png              # Convergence + Gantt + ISA vs Greedy
    ├── sa_dynamics.png          # Temperature cooling + targets assigned
    ├── isa_vs_greedy.png        # Bar chart at n=50, 75, 100
    ├── gamma_analysis.png       # Effect of γ parameter
    ├── mc_distribution.png      # Monte Carlo profit histogram
    └── resource_analysis.png    # Profit vs energy & memory limits
```

---

## Requirements

Python 3.8 or higher. Install dependencies with:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy matplotlib scipy
```

---

## How to Run

### Run the main scheduler (both ISA and Greedy baseline)

```bash
python main.py
```

Or using the legacy entry point:

```bash
python Satellite.py
```

This will:
1. Generate a synthetic problem with 80 targets and 2 satellites
2. Run the greedy baseline
3. Run the ISA algorithm (takes ~1–2 minutes)
4. Print final results to terminal
5. Save `results/results.png` and `results/sa_dynamics.png`

**Expected terminal output:**

```
============================================================
AEOS Scheduling
DA 243 — Introduction to Optimization
============================================================
Generating problem: 80 targets, 2 satellites
  Orbits: 28 | Visible (i,k) pairs: ...

Running Greedy Baseline...
  Greedy: XX targets, MC profit = XXX.XX

Running ISA...
  ...

============================================================
FINAL RESULTS
============================================================
  Greedy : XX targets | MC 90%-confidence profit = XXX.XX
  ISA    : XX targets | MC 90%-confidence profit = XXX.XX
  ISA improvement over Greedy: +XX.X%
```

### Generate all 4 comparison plots

```bash
python comparison_plots.py
```

This runs ISA 13 times across different configurations and saves 4 PNG files to `results/`. Takes approximately 5–10 minutes. To make it faster, open `plots/config.py` and reduce `N_FM = 50` to `N_FM = 30`.

---

## Key Parameters

Default ISA parameters are set in `main.py`. Comparison plot settings are in `plots/config.py`.

| Parameter      | Default | Meaning |
|----------------|---------|---------|
| `n_targets`    | 80      | Number of ground targets |
| `n_satellites` | 2       | Number of satellites (each makes 14 orbits/day) |
| `T0`           | 100.0   | Initial SA temperature |
| `alpha_T`      | 0.95    | Cooling rate — temperature drops 5% per outer loop |
| `alpha_L`      | 1.05    | Chain growth — inner loop grows 5% per outer loop |
| `gamma`        | 0.10    | Disturbance rate — delete 10% of targets each iteration |
| `n_Fm`         | 80      | Maximum outer loop iterations |
| `n_scenarios`  | 20      | Monte Carlo scenarios per evaluation (increase for more accuracy) |
| `DEFAULT_SEED` | 42      | Random seed for reproducibility (`aeos/config.py`) |

---

## Module Guide

| Module | Responsibility |
|--------|----------------|
| `aeos/data_generation.py` | Creates synthetic targets, orbits, visibility windows, cloud probabilities |
| `aeos/physics.py` | Attitude rotation time (Eq. 3), slew energy (Eq. 4), observation start (Eq. 1) |
| `aeos/constraints.py` | Checks all five feasibility constraints before assignment |
| `aeos/objective.py` | Monte Carlo 90%-confidence profit and fast deterministic proxy |
| `aeos/selection.py` | Target urgency Needᵢ (Eq. 21) and orbit conflict CFᵢₖ (Eq. 22) |
| `aeos/greedy.py` | Greedy baseline and ISA initial solution construction |
| `aeos/perturbation.py` | Delete γ% of targets and greedily reinsert |
| `aeos/isa.py` | Full ISA loop with Metropolis acceptance (Eq. 18) |
| `aeos/visualization.py` | Convergence, Gantt, and comparison plots |
| `plots/comparison.py` | Parameter sensitivity and resource analysis plots |

---

## Paper Equations Implemented

| Function                       | Equation  | Description |
|--------------------------------|-----------|-------------|
| `compute_trans(i, j, k)`       | Eq. 3     | Rotation time between consecutive targets |
| `compute_slew_energy(i, j, k)` | Eq. 4     | Energy consumed during attitude maneuver |
| `compute_obs_start(tp, i, k)`  | Eq. 1     | Observation start time from TPik variable |
| `check_feasibility()`          | Eq. 6–10  | All 5 constraints: C1 once-only, C2 visibility, C3 memory, C4 energy, C5 rotation |
| `monte_carlo_profit()`         | Eq. 15–16 | Sample approximation — 10th percentile of scenario profits |
| `compute_need(i)`              | Eq. 21    | Target urgency: Needᵢ = ωᵢ/ωₘₐₓ + avg(pᵢₖ)/Nᵢ |
| `compute_conflict(i, k)`       | Eq. 22    | Orbit conflict: CFᵢₖ = (1−pᵢₖ) × (overlap + resource fractions) |
| `run_isa()`                    | Eq. 18    | Metropolis acceptance: Pₐ = exp(Δf/T) if Δf < 0 |

---

## How the Algorithm Works

```
1. Generate synthetic problem (since the paper also used synthetic data)
   └── targets (profit, location, observation time)
   └── orbits  (energy capacity, memory capacity)
   └── visibility windows (VTSᵢₖ, VTEᵢₖ)
   └── cloud probabilities (pᵢₖ)

2. Greedy baseline
   └── Rank targets by Needᵢ (Eq. 21)
   └── Assign each to lowest-conflict orbit (CFᵢₖ, Eq. 22)
   └── Evaluate with Monte Carlo (Eq. 15–16)

3. ISA main loop
   └── OUTER LOOP: cool temperature T = T × 0.95
       └── INNER LOOP: Markov chain at fixed T
           └── Perturb: delete γ=10% of targets randomly
           └── Rebuild: reinsert by Need/CF priority
           └── Evaluate: deterministic expected profit (fast proxy)
           └── Accept: always if better, with probability exp(Δf/T) if worse [Eq. 18]
           └── Track: best schedule seen at any point

4. Output
   └── Best schedule found
   └── f = profit guaranteed in 90% of cloud scenarios (Monte Carlo)
   └── Plots showing convergence and comparison
```

---

## Skills Demonstrated

- **Combinatorial optimization** — NP-hard satellite scheduling with multiple coupled constraints
- **Stochastic programming** — Chance Constrained Programming via Monte Carlo sampling
- **Metaheuristics** — Simulated Annealing with adaptive cooling and chain-length growth
- **Python software design** — Modular package structure with clear separation of concerns
- **Scientific reproducibility** — Fixed random seeds, documented parameters, synthetic benchmark generation

---

## Why Synthetic Data?

The paper itself states:

> *"Without benchmark dataset for uncertain AEOSs scheduling problems, we design several instances..."* — Han et al., Section V-A

No public benchmark dataset exists for this problem. We follow the same distributions described in the paper: profits ~ Uniform[1, 10], cloud probabilities ~ Uniform[0, 1], and time windows sized realistically for agile satellites (8–20 minutes).

---

## Results Summary

Our implementation produces results qualitatively consistent with the paper. ISA consistently outperforms the greedy baseline across all tested instance sizes.

| Instance      | Greedy profit f | ISA profit f | Improvement |
|---------------|-----------------|--------------|-------------|
| n=50 targets  | 87.3            | 123.9        | 41.9%       |
| n=75 targets  | 154.2           | 191.3        | 24.1%       |
| n=100 targets | 195.0           | 290.5        | 49.0%       |

*Results will vary based on random seed and ISA parameters.*

**Key finding:** γ=0.10 (deleting 10% of targets per perturbation) gives the best balance between exploration and exploitation, consistent with the paper's conclusion.

---

## Limitations

- **Synthetic orbital data:** We use random visibility windows instead of real orbital mechanics (SGP4/TLE). The paper also uses synthetic data.
- **Smaller scale:** We test n=50–100 targets with 2 satellites. The paper uses n=500–950 with 4 satellites.
- **Simplified TPᵢₖ selection:** We try discrete values {0.1, 0.3, 0.5, 0.7, 0.9} instead of a full SQP optimizer.
- **Independent cloud model:** We assume independent Bernoulli cloud events per target. Real clouds are spatially correlated.

---

## Paper Reference

> Han, C., Gu, Y., Wu, G., & Wang, X. (2023). *Simulated Annealing Based Heuristic for Multiple Agile Satellites Scheduling under Cloud Coverage Uncertainty.* IEEE Transactions on Systems, Man, and Cybernetics: Systems.
