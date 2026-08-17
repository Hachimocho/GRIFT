"""Prune the training graph by predicted I-value, during training.

Rewritten after measuring the previous version on a real sweep, where it did nothing at
all: every update logged `42 weak / 0 strong` and `+0 / -0 edges`. Four independent
reasons, all now addressed.

**Absolute thresholds did not bracket the distribution.** "Weak" meant an I-value above
0.8 and "strong" below 0.2, and *no* node ever fell below 0.2. Since the old rewiring added
edges from weak nodes to strong ones, an empty strong set meant nothing was ever added, and
with nothing classified strong nothing was removed either. Thresholds are now **quantiles**
of the observed values, so both sets are non-empty by construction whatever scale the DQN
happens to output.

**The graph was far too dense for edge rewiring to matter.** The measured training graph
held 823,814 edges over 1,304 nodes -- an average degree of 1,264, meaning every node was
already connected to ~97% of the graph. Adding or dropping tens of edges is ~0.01% of the
topology, and `max_edges_per_node=10` was so far below the actual degree that
`add_edges_to_weak_node` returned immediately every time. So the unit of change is now the
**node**: a node the DQN considers uninformative is withdrawn from the training graph, which
alters what the traversal can reach regardless of density.

**It ran at most a handful of times.** The interval was counted in training steps and ticked
once per epoch, so only the epochs after the best checkpoint could matter. Updates are now
requested several times per epoch.

**Most nodes were never measured.** 300 of 1,304 were sampled per epoch, so the rest sat at
the neutral default and could not be classified either way. Performance is now recorded
opportunistically from the I-values training already computes, at O(1) memory per node --
see `track_performance`.

Removal is reversible: `restore_nodes` puts back the most recently withdrawn nodes, so a
drop in validation quality is recoverable without restarting.
"""

import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional  # noqa: F401 - used in annotations

from managers.GraphManager import GraphManager

#: Exponential-moving-average weight for a node's newest I-value.
#:
#: An EWMA rather than the previous list of the last 100 observations. That list cost roughly
#: 36 bytes per observation plus per-list overhead, so tracking a million nodes at full depth
#: would have needed several gigabytes; two floats per node is ~50 MB for the same coverage.
#: It also removes the arbitrary 100-sample horizon -- older evidence decays smoothly instead
#: of falling off a cliff.
EWMA_ALPHA = 0.3

#: Neutral I-value for a node nothing has measured yet. Matches the old default so an
#: untracked node is still classified as neither weak nor strong.
NEUTRAL_I_VALUE = 0.5

#: Minimum number of measured nodes before quantiles mean anything.
MIN_TRACKED_FOR_QUANTILES = 20

#: Never withdraw more than this fraction of the *current* graph in one update, and never
#: shrink it below `MIN_GRAPH_FRACTION` of where it started. Pruning that runs away leaves a
#: graph too small to train on, and the traversal's behavior on a starved graph is a much
#: bigger effect than the pruning being studied.
MAX_REMOVAL_FRACTION = 0.05
MIN_GRAPH_FRACTION = 0.5


class PerformanceGraphManager(GraphManager):
    """Withdraws uninformative nodes from the training graph as the DQN learns."""

    tags = ["performance", "i-value"]
    hyperparameters = {
        "parameters": {
            "weak_quantile": {"distribution": "uniform", "min": 0.7, "max": 0.95},
            "strong_quantile": {"distribution": "uniform", "min": 0.05, "max": 0.3},
            "removal_fraction": {"distribution": "uniform", "min": 0.01, "max": 0.05},
            "updates_per_epoch": {"distribution": "int_uniform", "min": 1, "max": 8},
        }
    }

    def __init__(self, graph, weak_quantile=0.9, strong_quantile=0.1,
                 removal_fraction=0.02, updates_per_epoch=4, max_edges_per_node=None,
                 remove_target='strong', ewma_alpha=EWMA_ALPHA,
                 # Retired names, accepted so an existing config still constructs.
                 rewire_threshold=None, edge_removal_threshold=None,
                 update_interval=None):
        """
        Args:
            weak_quantile: I-value quantile above which a node counts as weak -- the DQN
                expects to keep learning from it.
            strong_quantile: quantile below which a node counts as strong -- already learned,
                so the cheapest thing to withdraw.
            removal_fraction: share of the current graph withdrawn per update, capped by
                `MAX_REMOVAL_FRACTION`.
            updates_per_epoch: how many times per epoch the trainer should tick this.
            remove_target: `'strong'` withdraws already-learned nodes (curriculum pruning);
                `'weak'` withdraws the ones the model keeps failing on (noise pruning). Both
                are defensible research positions, which is why it is a knob.
            rewire_threshold / edge_removal_threshold / update_interval: the previous
                absolute thresholds. Accepted and ignored, with a notice, because they are
                not translatable -- they addressed edges, and this addresses nodes.
        """
        super().__init__(graph)
        for name, value in (("rewire_threshold", rewire_threshold),
                            ("edge_removal_threshold", edge_removal_threshold),
                            ("update_interval", update_interval)):
            if value is not None:
                print(f"[PerformanceGraphManager] {name}={value} is ignored: absolute "
                      f"I-value thresholds are replaced by quantiles, and edge rewiring by "
                      f"node removal. See --weak-quantile / --strong-quantile / "
                      f"--removal-fraction / --graph-updates-per-epoch.")

        self.weak_quantile = weak_quantile
        self.strong_quantile = strong_quantile
        self.removal_fraction = removal_fraction
        self.updates_per_epoch = max(1, int(updates_per_epoch))
        self.remove_target = remove_target
        self.ewma_alpha = ewma_alpha
        self.max_edges_per_node = max_edges_per_node

        # node_id -> (ewma, observation count). Keyed by id rather than by the node object
        # so the mapping survives a node leaving and re-entering the graph, and so its size
        # does not pin node objects alive.
        self.node_performance: "OrderedDict[str, tuple]" = OrderedDict()
        self.i_value_predictor = None

        self.initial_node_count = len(list(graph.get_nodes())) if graph else 0
        #: Withdrawn nodes, most recent last, so restoration is LIFO.
        self.removed_nodes: List = []
        self.update_history: List[dict] = []
        self.updates = 0

    # -- measurement ------------------------------------------------------------ #

    def set_i_value_predictor(self, predictor):
        """Record that predicted I-values are available. Without one they are noise."""
        self.i_value_predictor = predictor

    def track_performance(self, node, i_value):
        """Fold one observation into a node's running average. O(1) time and memory.

        Cheap enough to call for **every** node the traversal visits, which is the intended
        source: training already computes an I-value per visited node, so recording it here
        costs nothing extra and coverage grows with training rather than with a separate
        sampling pass. That is what makes "track all nodes" affordable -- the binding
        constraint was never the bookkeeping, it was the one DQN forward pass per node, and
        this reuses passes that already happened.
        """
        try:
            value = float(i_value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(value):
            return

        key = getattr(node, 'node_id', None)
        if key is None:
            return
        previous = self.node_performance.get(key)
        if previous is None:
            self.node_performance[key] = (value, 1)
        else:
            mean, count = previous
            self.node_performance[key] = (
                (1.0 - self.ewma_alpha) * mean + self.ewma_alpha * value, count + 1,
            )

    def get_node_avg_performance(self, node) -> float:
        """A node's running I-value, or the neutral default if never measured."""
        entry = self.node_performance.get(getattr(node, 'node_id', None))
        return entry[0] if entry else NEUTRAL_I_VALUE

    def tracked_count(self) -> int:
        return len(self.node_performance)

    def quantiles(self):
        """`(strong_cut, weak_cut)` from the measured values, or None if too few.

        Computed over observations rather than over the whole graph, so a node nothing has
        measured cannot drag the cut toward the neutral default.
        """
        if len(self.node_performance) < MIN_TRACKED_FOR_QUANTILES:
            return None
        values = np.fromiter(
            (mean for mean, _count in self.node_performance.values()),
            dtype=float, count=len(self.node_performance),
        )
        strong_cut = float(np.quantile(values, self.strong_quantile))
        weak_cut = float(np.quantile(values, self.weak_quantile))
        return strong_cut, weak_cut

    def identify_weak_nodes(self) -> List:
        """Nodes in the top `weak_quantile` of observed I-value."""
        cuts = self.quantiles()
        if cuts is None:
            return []
        _strong_cut, weak_cut = cuts
        return [
            node for node in self.graph.get_nodes()
            if self.get_node_avg_performance(node) >= weak_cut
        ]

    def identify_strong_nodes(self) -> List:
        """Nodes in the bottom `strong_quantile` of observed I-value."""
        cuts = self.quantiles()
        if cuts is None:
            return []
        strong_cut, _weak_cut = cuts
        return [
            node for node in self.graph.get_nodes()
            if self.get_node_avg_performance(node) <= strong_cut
        ]

    # -- mutation --------------------------------------------------------------- #

    def steps_between_updates(self, steps_per_epoch):
        """Training steps between updates, from `updates_per_epoch`."""
        return max(1, int(steps_per_epoch) // self.updates_per_epoch)

    def update_graph(self, steps_taken=1):
        """Withdraw a slice of the graph. Returns the update's stats, or None if it did not.

        `steps_taken` is accepted for the `GraphManager` contract and ignored: the trainer
        now decides when to tick this, several times per epoch, rather than the manager
        counting steps it cannot see.
        """
        if self.i_value_predictor is None:
            print("[PerformanceGraphManager] no I-value predictor set; skipping update. "
                  "Pruning needs an i-value traversal to supply one.")
            return None

        cuts = self.quantiles()
        if cuts is None:
            print(f"[PerformanceGraphManager] only {self.tracked_count()} node(s) measured "
                  f"(need {MIN_TRACKED_FOR_QUANTILES}); skipping update.")
            return None
        strong_cut, weak_cut = cuts

        current = list(self.graph.get_nodes())
        floor = int(self.initial_node_count * MIN_GRAPH_FRACTION)
        if len(current) <= floor:
            print(f"[PerformanceGraphManager] graph is at its floor "
                  f"({len(current)} <= {floor} nodes); skipping update.")
            return None

        candidates = (
            self.identify_strong_nodes() if self.remove_target == 'strong'
            else self.identify_weak_nodes()
        )
        budget = min(
            int(len(current) * min(self.removal_fraction, MAX_REMOVAL_FRACTION)),
            len(current) - floor,
        )
        # Worst-first within the candidate set, so a partial budget removes the clearest
        # cases. Sorted by node_id as the tie-break: `get_nodes` order is not guaranteed
        # stable across processes, and an unstable tie-break would make the choice depend on
        # PYTHONHASHSEED.
        candidates.sort(key=lambda node: (
            self.get_node_avg_performance(node) if self.remove_target == 'strong'
            else -self.get_node_avg_performance(node),
            str(getattr(node, 'node_id', '')),
        ))
        doomed = candidates[:max(0, budget)]

        # One batch pass. Removing node-by-node would be a linear index scan each time --
        # O(N*k), which at a million nodes is not runnable.
        removed = self.graph.remove_nodes(doomed) if doomed else 0
        self.removed_nodes.extend(doomed[:removed])

        self.updates += 1
        stats = {
            "update": self.updates,
            "tracked_nodes": self.tracked_count(),
            "graph_nodes": len(list(self.graph.get_nodes())),
            "strong_cut": strong_cut,
            "weak_cut": weak_cut,
            "weak_nodes": len(self.identify_weak_nodes()),
            "strong_nodes": len(self.identify_strong_nodes()),
            "nodes_removed": removed,
            "removed_total": len(self.removed_nodes),
            "remove_target": self.remove_target,
        }
        self.update_history.append(stats)
        print(f"[PerformanceGraphManager] update {self.updates}: "
              f"tracked {stats['tracked_nodes']}, cuts [{strong_cut:.4f}, {weak_cut:.4f}], "
              f"withdrew {removed} {self.remove_target} node(s), "
              f"graph now {stats['graph_nodes']} ({len(self.removed_nodes)} withdrawn)")
        return stats

    def restore_nodes(self, count=None):
        """Put back the most recently withdrawn nodes. Returns how many came back."""
        if not self.removed_nodes:
            return 0
        count = len(self.removed_nodes) if count is None else min(count, len(self.removed_nodes))
        restored = 0
        for _ in range(count):
            node = self.removed_nodes.pop()
            try:
                self.graph.add_node(node)
                restored += 1
            except Exception as error:
                print(f"[PerformanceGraphManager] could not restore "
                      f"{getattr(node, 'node_id', '?')}: {error}")
        if restored:
            print(f"[PerformanceGraphManager] restored {restored} node(s); graph now "
                  f"{len(list(self.graph.get_nodes()))}")
        return restored

    def get_stats(self):
        return {
            "updates": self.updates,
            "tracked_nodes": self.tracked_count(),
            "removed_total": len(self.removed_nodes),
            "graph_nodes": len(list(self.graph.get_nodes())) if self.graph else 0,
            "initial_node_count": self.initial_node_count,
            "history": list(self.update_history),
        }

