"""Per-node training state, so an I-value can depend on the model and not only the node.

`DQNCapability._get_dqn_features` feeds the DQN only *static* node attributes -- one-hot
demographics, quality scalars, the face embedding. Nothing in that vector changes as
training proceeds, so the learned ranking is a fixed per-sample difficulty prior: it cannot
notice that the model has since learned a sample. Informational value over a training run is
exactly that dynamic quantity, so with a static input the DQN cannot represent it however
good its target is.

This is the missing half of the input. It records what the model most recently did on each
node -- probability, loss, how often it has been trained on, how long ago -- from logits the
training loop has already computed. That matters for cost: the alternative, evaluating a
candidate node at selection time, would mean a detector forward pass per candidate per step,
which is precisely the cost that made the old full-graph I-value refresh unusable.

The EWMA shape is deliberately the same as `PerformanceGraphManager.node_performance`:
O(1) time and memory per node, folded in as observations arrive, coverage growing with
training rather than with a separate sampling pass.

Unseen nodes return fixed neutral features rather than being absent, because
`DQNCapability._initialize_dqns` probes `feature_dim` exactly once from `sample_nodes[0]` --
a variable-length vector would size the network off whichever node happened to be first.
"""

import math

#: Smoothing for the running estimates. Matches `PerformanceGraphManager.EWMA_ALPHA`.
EWMA_ALPHA = 0.3

#: Losses are divided by this and clipped to [0, 1] before entering the network. A raw BCE
#: is unbounded, and one badly-fit sample would otherwise dominate the input scale.
LOSS_SCALE = 5.0

#: Probability assumed for a node the model has never trained on: maximally uncommitted.
NEUTRAL_PROB = 0.5

#: Loss assumed for an unseen node -- the binary cross-entropy of a 0.5 prediction,
#: `-ln(0.5)`, which is what a model that knows nothing about a sample actually pays.
NEUTRAL_LOSS = math.log(2.0)

#: What an *unvisited* node's loss placeholder should be, relative to visited nodes.
#:
#: This is not a cosmetic default -- it silently decided the meaning of an entire experiment.
#: `NEUTRAL_LOSS / LOSS_SCALE` is 0.139, while a node that *has* been trained on carries its
#: real loss, whose median across 150,000 measured samples is **0.0008**. So the placeholder
#: outranks almost every visited node, and "select the highest I-value" quietly became
#: "prefer whatever you have not seen yet". The `loss_ewma` arm was built to test hard-sample
#: mining and instead ran as a novelty sampler: it trained on the easiest data in the sweep
#: (mean loss 0.116, 4.9% wrong, against 0.21-0.27 and 9-12% for the other arms) while
#: covering 134,096 unique nodes -- the broadest coverage and the worst detector.
#:
#: Making it explicit so an arm's name matches its behaviour:
#:   ``neutral``     -- 0.139, today's value. Prefers unvisited nodes, by accident.
#:   ``optimistic``  -- deliberately high. Prefers unvisited nodes, on purpose.
#:   ``pessimistic`` -- 0.0. An unvisited node ranks below any node with measured loss, so
#:                      the estimator has to earn a visit from the static features alone.
UNSEEN_PRIORS = ("neutral", "optimistic", "pessimistic")
DEFAULT_UNSEEN_PRIOR = "neutral"

#: The `optimistic` placeholder, in the same units as the scaled loss feature: above the
#: maximum any real loss can reach after clipping, so unvisited always wins.
OPTIMISTIC_LOSS = LOSS_SCALE * 1.0

#: `times_seen` saturates here. The difference between "seen once" and "seen twice" matters;
#: the difference between the 40th and 41st visit does not.
SEEN_SATURATION = 20.0

#: Staleness saturates here, in epochs.
STALENESS_SATURATION = 10.0

#: Index of the current-loss entry inside the vector `features` returns.
#:
#: Exported rather than left implicit because `GainEstimatorBase` initialises its direct
#: linear path with a positive weight on exactly this column -- that is what makes an
#: untrained model start out ranking by current loss, which is the baseline the learned
#: estimator has to beat. A silent reordering of `features` would point that initialisation
#: at the wrong signal, so the two are pinned together by a unit test.
STATE_LOSS_INDEX = 2

#: Length of the vector `features` returns. Asserted in tests: `feature_dim` is probed once,
#: so a silent change here would resize the DQN's input layer for every future run.
STATE_FEATURE_COUNT = 6


class NodeTrainingState:
    """Running per-node record of how the model currently handles each sample.

    Keyed by `node_id` rather than by node object: nodes are re-created when a graph is
    rebuilt from cache, and identity would not survive that while the id does.
    """

    def __init__(self, ewma_alpha=EWMA_ALPHA, unseen_prior=DEFAULT_UNSEEN_PRIOR):
        self.ewma_alpha = float(ewma_alpha)
        if unseen_prior not in UNSEEN_PRIORS:
            raise ValueError(
                f"unknown unseen_prior {unseen_prior!r}; choose from "
                f"{', '.join(UNSEEN_PRIORS)}"
            )
        self.unseen_prior = unseen_prior
        # node_id -> (prob_ewma, loss_ewma, times_seen, last_epoch)
        self._state = {}
        # node_id -> (mean_measured_gain, visits)
        self._gain = {}

    @property
    def unseen_loss(self):
        """The loss placeholder for a node that has never been trained on."""
        if self.unseen_prior == "optimistic":
            return OPTIMISTIC_LOSS
        if self.unseen_prior == "pessimistic":
            return 0.0
        return NEUTRAL_LOSS

    def __len__(self):
        return len(self._state)

    def observe_gain(self, node, gain):
        """Fold one *measured* learning gain into a node's running mean. O(1).

        Separate from `observe` because gain is a different kind of quantity: it is measured
        *after* the update rather than before it, and it is what decides whether training on
        this node has been helping. 38% of trained samples show a negative measured gain, so
        there is a large population of nodes that made the model worse on themselves.
        """
        key = getattr(node, 'node_id', None)
        if key is None:
            return
        try:
            gain = float(gain)
        except (TypeError, ValueError):
            return
        if not math.isfinite(gain):
            return
        previous = self._gain.get(key)
        if previous is None:
            self._gain[key] = (gain, 1)
            return
        mean, count = previous
        # A plain running mean, not an EWMA: the question is "has this node helped on
        # average", where every visit should count equally, not "is it helping lately".
        self._gain[key] = (mean + (gain - mean) / (count + 1), count + 1)

    def gain_record(self, node):
        """`(mean_gain, visits)` for `node`, or None if never measured."""
        return self._gain.get(getattr(node, 'node_id', None))

    def is_harmful(self, node, min_visits=3):
        """True when this node's mean measured gain is negative over enough visits.

        `min_visits` matters: a single negative gain is unremarkable -- the target's
        distribution straddles zero with a median of +0.005 -- so acting on one observation
        would withdraw a third of the dataset at random.
        """
        record = self._gain.get(getattr(node, 'node_id', None))
        if record is None:
            return False
        mean, visits = record
        return visits >= int(min_visits) and mean < 0.0

    def observe(self, node, prob, loss, epoch=0):
        """Fold one observation into a node's running estimates. O(1).

        Silently ignores non-finite values: a diverged batch should not poison the features
        that steer sampling for the rest of the run.
        """
        key = getattr(node, 'node_id', None)
        if key is None:
            return
        try:
            prob = float(prob)
            loss = float(loss)
            epoch = int(epoch)
        except (TypeError, ValueError):
            return
        if not (math.isfinite(prob) and math.isfinite(loss)):
            return

        previous = self._state.get(key)
        if previous is None:
            self._state[key] = (prob, loss, 1, epoch)
            return

        prob_ewma, loss_ewma, times_seen, _ = previous
        alpha = self.ewma_alpha
        self._state[key] = (
            (1.0 - alpha) * prob_ewma + alpha * prob,
            (1.0 - alpha) * loss_ewma + alpha * loss,
            times_seen + 1,
            epoch,
        )

    def get(self, node):
        """Raw `(prob, loss, times_seen, last_epoch)`, or None if never observed."""
        return self._state.get(getattr(node, 'node_id', None))

    def features(self, node, epoch=0):
        """Fixed-length model-state features for `node`, always `STATE_FEATURE_COUNT` long.

        Every element is bounded, so the vector cannot dominate the static attributes it is
        concatenated with.
        """
        record = self._state.get(getattr(node, 'node_id', None))
        if record is None:
            prob, loss, times_seen, last_epoch = NEUTRAL_PROB, self.unseen_loss, 0, epoch
            seen = 0.0
        else:
            prob, loss, times_seen, last_epoch = record
            seen = 1.0

        margin = abs(prob - 0.5) * 2.0
        staleness = max(0, int(epoch) - int(last_epoch)) / STALENESS_SATURATION
        return [
            seen,
            prob,
            min(max(loss, 0.0) / LOSS_SCALE, 1.0),
            margin,
            min(times_seen / SEEN_SATURATION, 1.0),
            min(staleness, 1.0),
        ]


__all__ = [
    "DEFAULT_UNSEEN_PRIOR", "OPTIMISTIC_LOSS", "UNSEEN_PRIORS",
    "EWMA_ALPHA", "LOSS_SCALE", "NEUTRAL_LOSS", "NEUTRAL_PROB", "NodeTrainingState",
    "STATE_LOSS_INDEX",
    "SEEN_SATURATION", "STALENESS_SATURATION", "STATE_FEATURE_COUNT",
]
