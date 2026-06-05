"""Synthetic AEOS problem instance generation."""

import random


# DATA GENERATION

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
