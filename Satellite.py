"""
=============================================================================
AEOS Scheduler
Simulated Annealing Based Scheduling of Agile Earth Observation Satellites
Under Cloud Coverage Uncertainty
=============================================================================
Based on: Han et al., IEEE TSMC: Systems
Course:   DA 243 — Introduction to Optimization


"We implemented the core ideas from the paper in Python. We simplified
orbital mechanics (using random VTWs instead of real orbits) but kept
the optimization structure identical: binary assignment variables xik,
continuous time-position variables TPik, energy/memory constraints,
Monte Carlo cloud evaluation, and the ISA(Improved Simulated Annealing) loop with Metropolis acceptance."
=============================================================================
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import time


np.random.seed(42)
random.seed(42)


#DATA GENERATION

# "We generate synthetic data because the paper has no public benchmark dataset.
#  We follow the same distributions described in Section V-A of the paper:
#  profits uniform in [1,10], cloud probabilities uniform in [0,1],
#  and time windows sized realistically for agile satellites."

def generate_problem(n_targets=80, n_satellites=2, horizon=1440):
    """
    Generate a synthetic AEOS scheduling problem instance.

    Parameters
    ----------
    n_targets    : number of ground targets to observe
    n_satellites : number of satellites (each makes ~14 orbits in 24hr)
    horizon      : planning horizon in minutes (1440 = 24 hours)

    Returns a dictionary with all problem data.
    """
    print(f"Generating problem: {n_targets} targets, {n_satellites} satellites")

    # --- Targets ---
    targets = []
    for i in range(n_targets):
        targets.append({
            'id':     i,
            'profit': round(random.uniform(1, 10), 2),   # ωᵢ ~ Uniform[1,10]
            'obs_time': random.randint(5, 15),            # otᵢ in minutes
            'lat':    random.uniform(-60, 60),            # latitude
            'lon':    random.uniform(-180, 180),          # longitude
        })

    # --- Orbits (each satellite does ~14 passes in 24 hours) ---

    # "Real VTWs require orbital propagators like SGP4. We approximate by
    #  randomly generating windows that mimic realistic satellite geometry:
    #  each target is visible on 1-4 orbits per satellite, each window is
    #  2-8 minutes long — consistent with real agile satellite parameters."
    orbits = []
    orbit_id = 0
    for sat in range(n_satellites):
        n_orbits = 14  # ~14 orbits per satellite in 24 hours
        for orb in range(n_orbits):
            # Orbit start time spread across the day
            orb_center = (orb / n_orbits) * horizon
            orbits.append({
                'id':       orbit_id,
                'sat_id':   sat,
                'center':   orb_center,
                # Resource budgets per orbit
                'E_cap':    80000,    # Eₖ = 80 kJ (in joules)
                'M_cap':    7500,     # Mₖ = 7500 MB
                'e_rate':   500,      # eₖ = 500 J/min imaging energy rate
                'm_rate':   100,      # mₖ = 100 MB/min memory rate
                'e_prime':  1000,     # e'ₖ = 1000 J/degree maneuvering rate
                'pitch_speed': 3,     # sᴾⁱᵗᶜʰ = 3 deg/min
                'roll_speed':  3,     # sᴿᵒˡˡ = 3 deg/min
            })
            orbit_id += 1

    n_orbits_total = len(orbits)

    # --- Visibility: which targets are visible on which orbits ---
    # bᵢₖ = 1 if target i is visible on orbit k
    # VTSᵢₖ, VTEᵢₖ: start and end of visibility window
    visibility = {}   # (i, k) -> True/False
    vtw = {}          # (i, k) -> (VTS, VTE)
    pik = {}          # (i, k) -> cloud-free probability

    for i, tgt in enumerate(targets):
        for k, orb in enumerate(orbits):
            # Each target is visible on ~25% of orbits(assumption)
            if random.random() < 0.25:
                # Window starts somewhere around orbit center
                window_start = orb['center'] + random.uniform(-20, 20)
                window_start = max(0, min(window_start, horizon - 20))
                window_len   = random.uniform(8, 20)   # 8-20 minute window
                window_end   = min(window_start + window_len, horizon)

                # Only valid if window is long enough to fit obs_time
                if window_end - window_start > tgt['obs_time']:
                    visibility[(i, k)] = True
                    vtw[(i, k)]        = (window_start, window_end)
                    pik[(i, k)]        = round(random.uniform(0, 1), 3)
                else:
                    visibility[(i, k)] = False
            else:
                visibility[(i, k)] = False

    # --- Observation angles for each visible (target, orbit) pair ---
    # θᴾⁱᵗᶜʰᵢₖ and θᴿᵒˡˡᵢₖ — bounded at ±30 degrees
    angles = {}
    for (i, k), vis in visibility.items():
        if vis:
            angles[(i, k)] = {
                'pitch': random.uniform(-30, 30),
                'roll':  random.uniform(-30, 30),
            }

    print(f"  Orbits: {n_orbits_total} | "
          f"Visible (i,k) pairs: {sum(visibility.values())}")

    return {
        'targets':    targets,
        'orbits':     orbits,
        'visibility': visibility,
        'vtw':        vtw,
        'pik':        pik,
        'angles':     angles,
        'n_targets':  n_targets,
        'n_orbits':   n_orbits_total,
        'horizon':    horizon,
    }



#  PHYSICS — Trans(i, j, k)

#  After observing target i, the satellite must
#  physically rotate its body to point at target j. The time required depends
#  on the angular difference in both pitch and roll axes, plus a stabilization
#  period. This is what couples the timing of consecutive observations —
#  choosing WHEN to observe i (via TPᵢₖ) changes the pointing angle,
#  which changes how long it takes to rotate to j."

def compute_trans(i, k, j, data, tp_i=0.5, tp_j=0.5):
    """
    Compute attitude transformation time from target i to j on orbit k.


    Returns time in minutes.
    """
    ang_i = data['angles'].get((i, k))
    ang_j = data['angles'].get((j, k))

    if ang_i is None or ang_j is None:
        return 999  # Not visible — infinite cost

    orb = data['orbits'][k]

    d_pitch = abs(ang_i['pitch'] - ang_j['pitch'])
    d_roll  = abs(ang_i['roll']  - ang_j['roll'])

    # Rotation time = max(pitch_time, roll_time)
    rot_time = max(d_pitch / orb['pitch_speed'],
                   d_roll  / orb['roll_speed'])

    # Stabilization time based on total angle change Δg
    delta_g = d_pitch + d_roll
    if delta_g <= 15:
        stab_time = 5 / 60    # 5 seconds in minutes
    elif delta_g <= 40:
        stab_time = 10 / 60
    else:
        stab_time = 15 / 60

    return rot_time + stab_time


def compute_slew_energy(i, k, j, data):
    """
    Compute maneuvering energy seᵢⱼₖ
    Energy = total angle change × energy rate e'ₖ
    """
    ang_i = data['angles'].get((i, k))
    ang_j = data['angles'].get((j, k))

    if ang_i is None or ang_j is None:
        return 0

    orb = data['orbits'][k]
    d_pitch = abs(ang_i['pitch'] - ang_j['pitch'])
    d_roll  = abs(ang_i['roll']  - ang_j['roll'])

    return (d_pitch + d_roll) * orb['e_prime']


def compute_obs_start(tp_ik, i, k, data):
    """
    Compute actual observation start time OTSᵢₖ from Equation 1.
    OTSᵢₖ = TPᵢₖ x (VTEᵢₖ - otᵢ VTSᵢₖ) + VTSᵢₖ
    """
    vts, vte = data['vtw'][(i, k)]
    ot = data['targets'][i]['obs_time']
    return tp_ik * (vte - ot - vts) + vts



# SOLUTION REPRESENTATION

# "A 'schedule' is represented as a dictionary mapping orbit k to an ordered
#  list of (target_id, TPᵢₖ) pairs. This directly encodes both decision
#  variables: xᵢₖ (a target appears in orbit k's list ↔ xᵢₖ=1) and
#  TPᵢₖ (stored alongside). We sort each orbit's targets by observation
#  start time to get a feasible ordered sequence."

class Schedule:
    """
    schedule[k] = list of (target_id, TPᵢₖ) tuples, ordered by start time
    """

    def __init__(self):
        self.assignment = defaultdict(list)  # k -> [(i, tp), ...]
        self.assigned_targets = set()        # set of assigned target ids

    def copy(self):
        new_s = Schedule()
        for k, seq in self.assignment.items():
            new_s.assignment[k] = list(seq)
        new_s.assigned_targets = set(self.assigned_targets)
        return new_s

    def add_target(self, i, k, tp):
        """Add target i to orbit k with time position tp."""
        self.assignment[k].append((i, tp))
        self.assigned_targets.add(i)
        # Keep sorted by observation start time
        self.assignment[k].sort(key=lambda x: x[1])

    def remove_target(self, i):
        """Remove target i from whatever orbit it's on."""
        for k in list(self.assignment.keys()):
            self.assignment[k] = [(t, tp) for t, tp in self.assignment[k] if t != i]
        self.assigned_targets.discard(i)

    def n_assigned(self):
        return len(self.assigned_targets)



#  FEASIBILITY CHECKING

# "Before accepting any target into the schedule we check all 5 constraints
#  from the paper. If any constraint is violated, xᵢₖ = 0 for that assignment.
#  The attitude transformation constraint (C5) is checked pairwise between
#  consecutive targets on the same orbit — this is where TPᵢₖ matters most."

def check_feasibility(i, k, tp_new, schedule, data):
    """
    Check if target i can be inserted into orbit k with time position tp_new.
    Returns (feasible: bool, reason: str)

    Checks constraints C1 through C5 from the paper.
    """
    orb  = data['orbits'][k]
    tgt  = data['targets'][i]
    vts, vte = data['vtw'][(i, k)]


    # C1: Each target observed at most once
    if i in schedule.assigned_targets:
        return False, "C1: already assigned"

    # C2: Target must be visible on this orbit
    if not data['visibility'].get((i, k), False):
        return False, "C2: not visible"

    # Compute actual start and end times for target i
    ots_i = compute_obs_start(tp_new, i, k, data)
    ote_i = ots_i + tgt['obs_time']

    # Must finish within the visibility window
    if ote_i > vte:
        return False, "C2: exceeds visibility window"

    # Get existing schedule on orbit k
    existing = schedule.assignment.get(k, [])

    # C5: Attitude transformation time with neighbours
    # Find predecessor and successor in the timeline
    pred = None  # (target_id, tp) of target just before i
    succ = None  # (target_id, tp) of target just after i

    for (j, tp_j) in existing:
        ots_j = compute_obs_start(tp_j, j, k, data)
        if ots_j < ots_i:
            if pred is None:
                pred = (j, tp_j)
            else:
                ots_pred = compute_obs_start(pred[1], pred[0], k, data)
                if ots_j > ots_pred:
                    pred = (j, tp_j)
        else:
            if succ is None:
                succ = (j, tp_j)
            else:
                ots_succ = compute_obs_start(succ[1], succ[0], k, data)
                if ots_j < ots_succ:
                    succ = (j, tp_j)

    # Check gap with predecessor
    if pred is not None:
        j, tp_j = pred
        ote_j   = compute_obs_start(tp_j, j, k, data) + data['targets'][j]['obs_time']
        trans   = compute_trans(j, k, i, data, tp_j, tp_new)
        if ots_i < ote_j + trans:
            return False, "C5: insufficient gap after predecessor"

    # Check gap with successor
    if succ is not None:
        j, tp_j = succ
        ots_j   = compute_obs_start(tp_j, j, k, data)
        trans   = compute_trans(i, k, j, data, tp_new, tp_j)
        if ots_j < ote_i + trans:
            return False, "C5: insufficient gap before successor"

    # C3 & C4: Resource constraints (check cumulative on orbit)
    # Memory used so far
    mem_used = sum(data['targets'][t]['obs_time'] * orb['m_rate']
                   for t, _ in existing)
    mem_new  = tgt['obs_time'] * orb['m_rate']
    if mem_used + mem_new > orb['M_cap']:
        return False, "C3: memory exceeded"

    # Energy used so far (imaging + maneuvering)
    e_imaging   = tgt['obs_time'] * orb['e_rate']
    e_maneuver  = 0
    if pred is not None:
        e_maneuver = compute_slew_energy(pred[0], k, i, data)

    e_used = sum(data['targets'][t]['obs_time'] * orb['e_rate']
                 for t, _ in existing)
    if e_used + e_imaging + e_maneuver > orb['E_cap']:
        return False, "C4: energy exceeded"

    return True, "OK"



#  OBJECTIVE FUNCTION — Monte Carlo Profit Evaluation
#  Instead of maximizing Σ ωᵢ·xᵢₖ (deterministic), we maximize the confidence
#  profit f = the profit achieved in at least 90% of cloud scenarios.
#  We implement this via Monte Carlo: sample n_scenarios random cloud
#  realizations, compute profit in each, and take the 10th percentile.
#  This directly approximates Eq.14: P(profit ≥ f) ≥ 1−α = 0.90."

def monte_carlo_profit(schedule, data, n_scenarios=30, alpha=0.10):
    """
    Evaluate schedule profit under cloud uncertainty.

    Implements the sample approximation from Section III:
    - Generate |W| = 30 cloud scenarios
    - In each scenario, λˡᵢₖ ~ Bernoulli(pᵢₖ)
    - Compute realized profit per scenario
    - Return the (alpha)-quantile as the confidence profit f


    "The 10th percentile (alpha=0.10) of scenario profits gives us f:
     the profit value exceeded in 90% of scenarios. This is our
     approximation of the CCP objective from Equation 13-14."
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


# SELECTION RULES — Needᵢ and CFᵢₖ

# "Before inserting targets we rank them by urgency and rank
#  orbits by conflict level. This greedy initialization gives
#  ISA a strong starting point — much better than random insertion."

def compute_need(i, data):
    """
    Compute urgency score Needᵢ from Equation 21.
    Needᵢ = ωᵢ/ωₘₐₓ + (Σₖ pᵢₖ) / Nᵢ
    Higher = schedule this target sooner.
    """
    omega_max = max(t['profit'] for t in data['targets'])
    omega_i   = data['targets'][i]['profit']

    # Average cloud-free probability across visible orbits
    visible_probs = [data['pik'][(i, k)]
                     for k in range(data['n_orbits'])
                     if data['visibility'].get((i, k), False)]

    if not visible_probs:
        return 0.0  # No visible orbits — can never be scheduled

    avg_prob = sum(visible_probs) / len(visible_probs)
    N_i      = len(visible_probs)

    return (omega_i / omega_max) + (sum(visible_probs) / N_i)


def compute_conflict(i, k, schedule, data):
    """
    Compute resource conflict score CFᵢₖ.
    CFᵢₖ = (1-pᵢₖ) x (time_overlap_fraction + resource_fraction)
    Lower = better orbit choice.
    """
    if not data['visibility'].get((i, k), False):
        return float('inf')

    orb      = data['orbits'][k]
    p_ik     = data['pik'].get((i, k), 0)
    vts_i, vte_i = data['vtw'][(i, k)]
    vtw_len  = vte_i - vts_i

    # Time overlap with already-scheduled targets on this orbit
    overlap = 0.0
    for (j, tp_j) in schedule.assignment.get(k, []):
        vts_j, vte_j = data['vtw'].get((j, k), (0, 0))
        # Overlap = intersection of two windows
        ol = max(0, min(vte_i, vte_j) - max(vts_i, vts_j))
        overlap += ol

    overlap_frac = overlap / vtw_len if vtw_len > 0 else 1.0

    # Resource fraction (memory)
    mem_used = sum(data['targets'][t]['obs_time'] * orb['m_rate']
                   for t, _ in schedule.assignment.get(k, []))
    mem_frac = mem_used / orb['M_cap'] if orb['M_cap'] > 0 else 1.0

    return (1 - p_ik) * (overlap_frac + mem_frac)



# GREEDY INITIAL SOLUTION

# "We build the initial solution by repeatedly picking the highest-Need
#  target and inserting it on its lowest-conflict orbit. This is the
#  'Stage 1' of the ISA heuristic described in Section IV-A of the paper.
#  A good initial solution speeds up convergence significantly."

def build_initial_solution(data):
    """
    Construct an initial feasible schedule using Need/CF selection rules.
    Returns a Schedule object.
    """
    schedule = Schedule()
    n = data['n_targets']

    # Rank all targets by urgency (highest Need first)
    needs = [(compute_need(i, data), i) for i in range(n)]
    needs.sort(reverse=True)

    inserted = 0
    for _, i in needs:
        # Find best orbit for this target (lowest CF)
        candidate_orbits = [
            (compute_conflict(i, k, schedule, data), k)
            for k in range(data['n_orbits'])
            if data['visibility'].get((i, k), False)
               and i not in schedule.assigned_targets
        ]

        if not candidate_orbits:
            continue

        candidate_orbits.sort()  # Sort by conflict score

        # Try to insert on best orbits
        for _, k in candidate_orbits[:3]:   # try top-3 orbit choices
            tp = 0.5  # Start at midpoint (best image quality, Eq.30)
            feasible, reason = check_feasibility(i, k, tp, schedule, data)
            if feasible:
                schedule.add_target(i, k, tp)
                inserted += 1
                break
            else:
                # Try a few other positions in the window
                for tp_try in [0.3, 0.7, 0.1, 0.9]:
                    feasible, _ = check_feasibility(i, k, tp_try, schedule, data)
                    if feasible:
                        schedule.add_target(i, k, tp_try)
                        inserted += 1
                        break
                if i in schedule.assigned_targets:
                    break

    print(f"  Initial solution: {inserted}/{n} targets scheduled")
    return schedule



# PERTURBATION — Delete and Reinsert


#  We randomly remove γ×|schedule|
#  targets and then try to reinsert them using the greedy insertion logic.
#  This is how ISA explores the solution space — by partially destroying
#  and rebuilding the schedule in a different way."

def perturb_and_reinsert(schedule, data, gamma=0.10):
    """
    Perturbation operator: delete γ fraction of assigned targets,
    then greedily reinsert them (and all previously unassigned targets).

    Returns a new Schedule.
    """
    new_schedule = schedule.copy()
    assigned = list(new_schedule.assigned_targets)

    if not assigned:
        return new_schedule

    # How many to delete
    n_delete = max(1, int(gamma * len(assigned)))
    to_delete = random.sample(assigned, n_delete)

    # Delete them
    for i in to_delete:
        new_schedule.remove_target(i)

    # Try to reinsert deleted targets plus any previously unscheduled ones
    unassigned = [i for i in range(data['n_targets'])
                  if i not in new_schedule.assigned_targets]

    # Rank by Need
    ranked = sorted(unassigned, key=lambda i: compute_need(i, data), reverse=True)

    for i in ranked:
        candidate_orbits = [
            (compute_conflict(i, k, new_schedule, data), k)
            for k in range(data['n_orbits'])
            if data['visibility'].get((i, k), False)
        ]
        candidate_orbits.sort()

        for _, k in candidate_orbits[:3]:
            for tp in [0.5, 0.3, 0.7, 0.1, 0.9]:
                feasible, _ = check_feasibility(i, k, tp, new_schedule, data)
                if feasible:
                    new_schedule.add_target(i, k, tp)
                    break
            if i in new_schedule.assigned_targets:
                break

    return new_schedule



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
    Returns (best_schedule, history) where history tracks profit over time.


    "Outer loop = temperature levels (gets colder each iteration).
     Inner loop = Markov chain (explore at this temperature).
     Each iteration: perturb → evaluate → Metropolis accept/reject.
     Track best solution seen at any point (Sbest, fbest)."
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



# GREEDY BASELINE (for comparison)

# "We compare ISA against a simple greedy algorithm that just inserts
#  targets in order of decreasing profit × probability, without any
#  optimization. This baseline shows how much ISA improves over a
#  naive approach — justifying the complexity of the SA framework."

def run_greedy(data):
    """
    Simple greedy baseline: insert targets sorted by ωᵢ x pᵢₖ (expected profit).
    No optimization — one pass, no perturbation.
    """
    schedule = Schedule()
    n = data['n_targets']

    # Sort by max expected profit across all visible orbits
    scores = []
    for i in range(n):
        max_ep = max(
            (data['targets'][i]['profit'] * data['pik'].get((i, k), 0)
             for k in range(data['n_orbits'])
             if data['visibility'].get((i, k), False)),
            default=0
        )
        scores.append((max_ep, i))

    scores.sort(reverse=True)

    for _, i in scores:
        for k in range(data['n_orbits']):
            if not data['visibility'].get((i, k), False):
                continue
            for tp in [0.5, 0.3, 0.7, 0.1, 0.9]:
                feasible, _ = check_feasibility(i, k, tp, schedule, data)
                if feasible:
                    schedule.add_target(i, k, tp)
                    break
            if i in schedule.assigned_targets:
                break

    f_greedy = monte_carlo_profit(schedule, data, n_scenarios=50, alpha=0.10)
    print(f"  Greedy: {schedule.n_assigned()} targets, "
          f"MC profit = {f_greedy:.2f}")
    return schedule, f_greedy



#  VISUALISATION


# "We produce three plots: (1) convergence curve showing how ISA profit
#  improves over iterations — this demonstrates the SA search working;
#  (2) schedule Gantt chart showing each orbit's observation timeline;
#  (3) profit comparison between ISA and greedy baseline."

def plot_results(history, schedule_isa, schedule_greedy,
                 f_isa_mc, f_greedy_mc, data):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("AEOS Scheduling under Cloud Coverage Uncertainty — ISA Results",
                 fontsize=13, fontweight='bold', y=1.02)

    #  Plot 1: Convergence curve
    ax1 = axes[0]
    ax1.plot(history['iteration'], history['profit'],
             color='#90A4AE', linewidth=1, label='Current profit', alpha=0.7)
    ax1.plot(history['iteration'], history['best_profit'],
             color='#1565C0', linewidth=2, label='Best profit found')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Deterministic Profit', fontsize=11)
    ax1.set_title('ISA Convergence', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.spines[['top', 'right']].set_visible(False)

    # Mark initial profit
    ax1.axhline(y=history['profit'][0], color='#E53935',
                linestyle='--', linewidth=1, alpha=0.6, label='Initial')

    #  Plot 2: Schedule Gantt chart
    ax2 = axes[1]
    colors = plt.cm.tab20.colors
    y_ticks, y_labels = [], []

    sat_colors = ['#1565C0', '#00897B', '#E65100', '#6A1B9A']

    for idx, (k, seq) in enumerate(sorted(schedule_isa.assignment.items())):
        if not seq:
            continue
        orb  = data['orbits'][k]
        sat  = orb['sat_id']
        col  = sat_colors[sat % len(sat_colors)]

        for (i, tp) in seq:
            ots = compute_obs_start(tp, i, k, data)
            ot  = data['targets'][i]['obs_time']
            p   = data['pik'].get((i, k), 0)
            # Alpha encodes cloud probability — clearer = more opaque
            alpha_val = 0.3 + 0.7 * p

            ax2.barh(idx, ot, left=ots, height=0.6,
                     color=col, alpha=alpha_val, edgecolor='white', linewidth=0.5)

        y_ticks.append(idx)
        y_labels.append(f"Orb {k}\n(Sat{sat})")

    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_labels, fontsize=7)
    ax2.set_xlabel('Time (minutes)', fontsize=11)
    ax2.set_title('Schedule Gantt Chart\n(opacity = cloud-free probability)',
                  fontsize=12, fontweight='bold')
    ax2.spines[['top', 'right']].set_visible(False)


    patches = [mpatches.Patch(color=sat_colors[s], label=f'Satellite {s}')
               for s in range(data['orbits'][-1]['sat_id'] + 1)]
    ax2.legend(handles=patches, fontsize=8, loc='upper right')

    #  Plot 3: ISA vs Greedy comparison
    ax3 = axes[2]
    methods  = ['Greedy\nBaseline', 'ISA\n(Proposed)']
    mc_profs = [f_greedy_mc, f_isa_mc]
    assigned = [schedule_greedy.n_assigned(), schedule_isa.n_assigned()]
    bar_cols = ['#EF5350', '#1565C0']

    bars = ax3.bar(methods, mc_profs, color=bar_cols,
                   width=0.5, edgecolor='white', linewidth=1.5)

    # Annotate bars
    for bar, prof, asgn in zip(bars, mc_profs, assigned):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{prof:.1f}\n({asgn} targets)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    improvement = ((f_isa_mc - f_greedy_mc) / max(f_greedy_mc, 1)) * 100
    ax3.set_ylabel('Monte Carlo 90%-Confidence Profit', fontsize=11)
    ax3.set_title(f'ISA vs Greedy Baseline\n(+{improvement:.1f}% improvement)',
                  fontsize=12, fontweight='bold')
    ax3.set_ylim(0, max(mc_profs) * 1.25)
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(r'D:\SEM4\Simulated_Annealing\results.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results.png")


def plot_temperature(history):
    """Plot temperature cooling curve to show SA annealing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Simulated Annealing Dynamics", fontsize=13, fontweight='bold')

    iters = history['iteration']

    # Temperature decay
    ax1.plot(iters, history['temperature'], color='#E65100', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Temperature T', fontsize=11)
    ax1.set_title('Temperature Cooling Schedule\nT ← T × αT (αT = 0.95)',
                  fontsize=11, fontweight='bold')
    ax1.fill_between(iters, history['temperature'],
                     alpha=0.15, color='#E65100')
    ax1.grid(True, alpha=0.3)
    ax1.spines[['top', 'right']].set_visible(False)

    # Targets assigned over time
    ax2.plot(iters, history['n_assigned'], color='#00897B', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Targets Scheduled', fontsize=11)
    ax2.set_title('Targets Assigned Over Time\n(shows exploration vs exploitation)',
                  fontsize=11, fontweight='bold')
    ax2.fill_between(iters, history['n_assigned'],
                     alpha=0.15, color='#00897B')
    ax2.grid(True, alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('D:\SEM4\Simulated_Annealing\sa_dynamics.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: sa_dynamics.png")



if __name__ == "__main__":
    print("="*60)
    print("AEOS Scheduling")
    print("DA 243 — Introduction to Optimization")
    print("="*60)

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
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
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

    
