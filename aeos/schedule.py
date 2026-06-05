"""Schedule data structure for target-orbit assignments."""

from collections import defaultdict


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
