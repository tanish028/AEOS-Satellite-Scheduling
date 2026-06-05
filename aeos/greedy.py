"""Greedy baseline and initial solution construction."""

from aeos.constraints import check_feasibility
from aeos.objective import monte_carlo_profit
from aeos.schedule import Schedule
from aeos.selection import compute_need, compute_conflict


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
