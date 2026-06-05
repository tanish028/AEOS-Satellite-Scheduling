"""Target urgency and orbit conflict scoring rules."""


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
