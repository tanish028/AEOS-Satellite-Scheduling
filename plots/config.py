"""Plot styling and experiment constants."""

import os

import matplotlib

from aeos.config import DEFAULT_SEED, RESULTS_DIR

SAVE_DIR = RESULTS_DIR

N_FM              = 50    # ISA outer iterations (reduce to 30 for faster run)
N_SCENARIOS_ISA   = 20    # MC scenarios used DURING ISA
N_SCENARIOS_FINAL = 93
SEED              = DEFAULT_SEED

BLUE  = "#2E75B6"
GRAY  = "#9E9E9E"
GREEN = "#1D7A4F"
RED   = "#C62828"

matplotlib.rcParams.update({
    "font.family":       "Arial",
    "font.size":         12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E5E5E5",
    "grid.linewidth":    0.8,
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "axes.edgecolor":    "#AAAAAA",
    "axes.linewidth":    0.8,
    "xtick.bottom":      False,
    "ytick.left":        False,
})
