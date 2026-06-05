"""Schedule perturbation operator for ISA exploration."""

import random

from aeos.constraints import check_feasibility
from aeos.selection import compute_need, compute_conflict


# PERTURBATION — Delete and Reinsert


#  We randomly remove γ×|schedule|
#  targets and then try to reinsert them using the greedy insertion logic.
#  This is how ISA explores the solution space — by partially destroying
#  and rebuilding the schedule in a different way.

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
