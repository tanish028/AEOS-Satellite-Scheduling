"""Result plots for ISA convergence and schedule comparison."""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from aeos.config import RESULTS_DIR
from aeos.physics import compute_obs_start


#  VISUALISATION


# "We produce three plots: (1) convergence curve showing how ISA profit
#  improves over iterations — this demonstrates the SA search working;
#  (2) schedule Gantt chart showing each orbit's observation timeline;
#  (3) profit comparison between ISA and greedy baseline."

def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_results(history, schedule_isa, schedule_greedy,
                 f_isa_mc, f_greedy_mc, data):

    _ensure_results_dir()

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
    path = os.path.join(RESULTS_DIR, "results.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_temperature(history):
    """Plot temperature cooling curve to show SA annealing."""
    _ensure_results_dir()

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
    path = os.path.join(RESULTS_DIR, "sa_dynamics.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
