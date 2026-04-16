# AEOS Satellite Scheduling under Cloud Coverage Uncertainty

Implementation of a Simulated Annealing based heuristic for scheduling Agile Earth Observation Satellites (AEOS) under cloud coverage uncertainty.

**Based on:** Han, Gu, Wu & Wang — *"Simulated Annealing Based Heuristic for Multiple Agile Satellites Scheduling under Cloud Coverage Uncertainty"* — IEEE Transactions on Systems, Man, and Cybernetics: Systems.

**Course:** DA 243 — Introduction to Optimization, 2026
**Institute:** Mehta Family School of Data Science and Artificial Intelligence, IIT Guwahati

---

## Team

| Name                 |Roll Number  |
|----------------------|-------------|
| Tanish Anand         | 240150037 |
| Tumu Sai Seshi Kiran | 240150038 |
| Rathod Ravi Kiran    | 240150029 |

---

## Project Overview

Agile Earth Observation Satellites (AEOS) can rotate in three directions (roll, pitch, yaw), giving them a flexible time window to observe each ground target. This flexibility makes scheduling much more powerful but also NP-hard.

The key challenge: **cloud coverage randomly blocks optical sensors**, so we cannot simply maximize deterministic profit. Instead, we use **Chance Constrained Programming (CCP)** to find schedules that achieve a guaranteed profit level with 90% confidence.

Our implementation follows as :
- Synthetic problem generation (targets, orbits, visibility windows, cloud probabilities)
- Greedy baseline scheduler
- Improved Simulated Annealing (ISA) with Metropolis(temparature-based) acceptance
- Monte Carlo profit evaluation under cloud uncertainty
- Result plots comparing ISA against the greedy baseline

---

## Repository Structure

```
AEOS-Satellite-Scheduling/
│
├── Satellite.py          # Main implementation — all algorithms
├── generate_plots.py     # Generates the 4 result plots
├── README.md
│
└── results/
    ├── results.png           # Convergence curve + Gantt chart + ISA vs Greedy
    ├── sa_dynamics.png       # Temperature cooling + targets assigned over time
    ├── isa_vs_greedy.png     # Bar chart: ISA vs Greedy at n=50, 75, 100
    ├── gamma_analysis.png    # Line chart: effect of gamma parameter
    ├── mc_distribution.png   # Histogram: Monte Carlo scenario profits
    └── resource_analysis.png # Bar chart: profit vs energy & memory limits
```

---

## Requirements

Python 3.8 or higher. Install dependencies with:

```bash
pip install numpy matplotlib scipy
```

## How to Run

### Run the main scheduler(both ISA and Grredy baseline)

```bash
python Satellite.py
```

This will:
1. Generate a synthetic problem with 80 targets and 2 satellites
2. Run the greedy baseline
3. Run the ISA algorithm (takes ~1–2 minutes)
4. Print final results to terminal
5. Save `results.png` and `sa_dynamics.png`

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

### Generate all 4 result plots

```bash
python comparison_plots.py
```

This runs ISA 13 times across different configurations and saves 4 PNG files. Takes approximately 5–10 minutes. To make it faster, open `comparison_plots.py` and reduce `N_FM = 50` to `N_FM = 30`.


## Key Parameters

These can be changed at the bottom of `Satellite.py` in the `run_isa()` call:

| Parameter      |Default|   Meaning |
|----------------|-------|-----------|
| `n_targets`    | 80    | Number of ground targets |
| `n_satellites` | 2     | Number of satellites (each makes 14 orbits/day) |
| `T0`           | 100.0 | Initial SA temperature |
| `alpha_T`      | 0.95  | Cooling rate — temperature drops 5% per outer loop |
| `alpha_L`      | 1.05  | Chain growth — inner loop grows 5% per outer loop |
| `gamma`        | 0.10  | Disturbance rate — delete 10% of targets each iteration |
| `n_Fm`         | 80    | Maximum outer loop iterations |
| `n_scenarios`  | 20    | Monte Carlo scenarios per evaluation (increase for more accuracy) |

---

## Paper Equations Implemented

| Function                       | Equation  | Description |
|--------------------------------|-----------|-------------|
| `compute_trans(i, j, k)`       | Eq. 3     | Rotation time between consecutive targets |
| `compute_slew_energy(i, j, k)` | Eq. 4     | Energy consumed during attitude maneuver |
| `compute_obs_start(tp, i, k)`  | Eq. 1     | Observation start time from TPik variable |
| `check_feasibility()`          | Eq. 6–10  | All 5 constraints: C1 once-only, C2 visibility, C3 memory, C4 energy, C5 rotation |
| `monte_carlo_profit()`         | Eq. 15–16 | Sample approximation — 10th percentile of scenario profits |
| `compute_need(i)`              | Eq. 21 | Target urgency: Needi = ωi/ωmax + avg(pik)/Ni |
| `compute_conflict(i, k)`       | Eq. 22 | Orbit conflict: CFik = (1−pik) × (overlap + resource fractions) |
| `run_isa()`                    | Eq. 18 | Metropolis acceptance: Pa = exp(Δf/T) if Δf < 0 |

---

## How the Algorithm Works

```
1. Generate synthetic problem(sinice the paper also used synthetic dta )
   └── targets (profit, location, observation time)
   └── orbits  (energy capacity, memory capacity)
   └── visibility windows (VTSik, VTEik)
   └── cloud probabilities (pik)

2. Greedy baseline
   └── Rank targets by Need_i (Eq.21)
   └── Assign each to lowest-conflict orbit (CF_ik, Eq.22)
   └── Evaluate with Monte Carlo (Eq.15-16)

3. ISA main loop
   └── OUTER LOOP: cool temperature T = T × 0.95
       └── INNER LOOP: Markov chain at fixed T
           └── Perturb: delete γ=10% of targets randomly
           └── Rebuild: reinsert by Need/CF priority
           └── Evaluate: Monte Carlo 90%-confidence profit f
           └── Accept: always if better, with probabilty exp(Δf/T) if worse [Eq.18]
           └── Track: best schedule seen at any point

4. Output
   └── Best schedule found
   └── f = profit guaranteed in 90% of cloud scenarios
   └── Plots showing convergence and comparison
```

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
| n=50 targets  | ~120            | ~170         | ~39% |
| n=75 targets  | ~190            | ~265         | ~39% |
| n=100 targets | ~245            | ~290         | ~18% |

*Exact values vary by run. Results will depend on randon seed chosen and ISA parameters.*

**Key finding:** γ=0.10 (deleting 10% of targets per perturbation) gives the best balance between exploration and exploitation, consistent with the paper's conclusion.

---

## Limitations

- **Synthetic orbital data:** We use random visibility windows instead of real orbital mechanics (SGP4/TLE). The paper also uses synthetic data.
- **Smaller scale:** We test n=50–100 targets with 2 satellites. The paper uses n=500–950 with 4 satellites.
- **Simplified TPik selection:** We try discrete values {0.1, 0.3, 0.5, 0.7, 0.9} instead of a full SQP optimizer.
- **Independent cloud model:** We assume independent Bernoulli cloud events per target. Real clouds are spatially correlated.

---

## Paper Reference

> Han, C., Gu, Y., Wu, G., & Wang, X. (2023). *Simulated Annealing Based Heuristic for Multiple Agile Satellites Scheduling under Cloud Coverage Uncertainty.* IEEE Transactions on Systems, Man, and Cybernetics: Systems.


