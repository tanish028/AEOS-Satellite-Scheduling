"""Feasibility checking against paper constraints C1–C5."""

from aeos.physics import compute_trans, compute_slew_energy, compute_obs_start


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
