"""Attitude maneuver and observation timing calculations."""


#  PHYSICS — Trans(i, j, k)

#  After observing target i, the satellite must
#  physically rotate its body to point at target j. The time required depends
#  on the angular difference in both pitch and roll axes, plus a stabilization
#  period. This is what couples the timing of consecutive observations —
#  choosing WHEN to observe i (via TPᵢₖ) changes the pointing angle,
#  which changes how long it takes to rotate to j.

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
