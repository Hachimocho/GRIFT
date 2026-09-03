"""Up-weight or preferentially select samples from the demographic group the model is
*currently* doing worst on, tracked from realised outcomes rather than predicted I-value.

`GroupTargeting` (`group_targeting.py`) already targets groups by mean I-value, on the
theory that a group the DQN rates as hard is a group worth training on. This is a
different, more direct signal: mean I-value is what the estimator *predicts* about a
group, which the rest of this project's own findings say is not very trustworthy (see
`docs/ivalue_gate_result.md` -- the DQN's own output barely correlates with realised
learning gain). Tracking realised per-group *error rate* instead needs no estimator at
all, costs nothing beyond what training already computes (`preds == labels` is summed
into `correct` every batch regardless), and answers the question directly: which group
is the model actually getting wrong right now.

One tracker, two uses, both already-established shapes elsewhere in this project:

* **Selection** (`is_targeted`): the same binary "is this node's group in the worst-K"
  filter `GroupTargeting` exposes, so it drops into `IValueTraversal`'s existing
  `group_targeting=` slot unchanged -- the two are interchangeable by construction (both
  answer "is this node's group targeted", just from different signals), never composed.
* **Weighting** (`multiplier`): a per-node scalar in `[1/clip, clip]`, geometric about 1
  like every other weight in `loss_weighting.py`, that composes multiplicatively with
  whatever `LossWeighter` already produces via its `extra_weights` parameter -- so a
  fairness-only run (`ivalue_loss_weight=none` + this) and a fairness-on-top-of-midband
  run use the exact same plumbing, not two.
"""

import math

from trainers.capabilities.group_targeting import MIN_GROUP_OBSERVATIONS, group_key

#: How many groups the multiplier considers "targeted" by `is_targeted` -- unused by
#: `multiplier`, which grades every eligible group continuously rather than picking a
#: fixed number of "worst" ones.
DEFAULT_TARGET_GROUPS = 3


class GroupPerformanceTracker:
    """Running per-group error rate, fed from realised (not predicted) outcomes."""

    def __init__(self, target_groups=DEFAULT_TARGET_GROUPS,
                 min_observations=MIN_GROUP_OBSERVATIONS, enabled=False):
        self.enabled = bool(enabled)
        self.target_groups = max(1, int(target_groups))
        self.min_observations = max(1, int(min_observations))
        self._means = {}       # group -> (mean_error, count)
        self._targeted = ()    # cached, recomputed when asked

    def observe(self, node, correct):
        """Fold one realised outcome into its group's running error rate. O(1).

        `correct` is whatever the caller already computed to increment its own running
        accuracy counter -- a bool, or anything `bool()` accepts (a 0/1 tensor element
        included), so this never requires a second correctness computation.
        """
        if not self.enabled:
            return
        key = group_key(node)
        if key is None:
            return
        error = 0.0 if bool(correct) else 1.0
        mean, count = self._means.get(key, (0.0, 0))
        self._means[key] = (mean + (error - mean) / (count + 1), count + 1)
        self._targeted = ()

    def _eligible(self):
        """`{group: mean_error}` for groups seen `min_observations` times or more."""
        return {
            key: mean for key, (mean, count) in self._means.items()
            if count >= self.min_observations
        }

    def targeted_groups(self):
        """The `target_groups` groups with the *highest* mean error, among those eligible.

        Mirrors `GroupTargeting.targeted_groups` exactly except for the sort direction
        (worst error, not highest I-value) -- see the module docstring for why the two
        are interchangeable at every call site that consumes `is_targeted`.
        """
        if self._targeted:
            return self._targeted
        eligible = self._eligible()
        if not eligible:
            return ()
        ranked = sorted(eligible.items(), key=lambda item: (-item[1], item[0]))
        self._targeted = tuple(key for key, _mean in ranked[:self.target_groups])
        return self._targeted

    def is_targeted(self, node):
        """True when `node`'s group is among the current worst-K by error.

        Same contract as `GroupTargeting.is_targeted`: true while nothing is eligible
        yet, so an early epoch (before `min_observations` is reached for any group)
        samples normally instead of rejecting every candidate and starving the walk.
        """
        if not self.enabled:
            return True
        targeted = self.targeted_groups()
        if not targeted:
            return True
        key = group_key(node)
        return key is None or key in targeted

    def multiplier(self, node, clip=2.0):
        """Bounded per-node weight: >1 for a worse-than-average group, <1 for better.

        `ratio = this group's error / mean error over every eligible group`, then
        clipped in *log* space so the bound is symmetric and geometric about 1 -- the
        same construction `ivalue_weights`'s `clip ** (2*scaled - 1)` uses, so a
        fairness multiplier and an I-value weight are on a directly comparable scale
        when `LossWeighter` multiplies them together.

        Returns 1.0 (a no-op factor) whenever there is nothing to grade against: the
        tracker is disabled, the node's group is unknown, or its group has not yet
        reached `min_observations` -- an under-observed group must not be boosted on
        the strength of a handful of noisy samples.
        """
        if not self.enabled:
            return 1.0
        eligible = self._eligible()
        if not eligible:
            return 1.0
        key = group_key(node)
        if key is None or key not in eligible:
            return 1.0
        overall = sum(eligible.values()) / len(eligible)
        if overall <= 1e-9:
            return 1.0
        ratio = eligible[key] / overall
        clip = max(1.0, float(clip))
        log_bound = math.log(clip)
        log_ratio = max(-log_bound, min(log_bound, math.log(max(ratio, 1e-6))))
        return math.exp(log_ratio)

    def summary(self):
        eligible = self._eligible()
        targeted = self.targeted_groups()
        return {
            "groups_seen": len(self._means),
            "groups_eligible": len(eligible),
            "targeted": list(targeted),
            "targeted_errors": [round(self._means[k][0], 4) for k in targeted],
            "overall_mean_error": (
                round(sum(eligible.values()) / len(eligible), 4) if eligible else None
            ),
        }


def fairness_weights_for_batch(nodes, tracker, clip=2.0, device=None):
    """Per-node fairness multipliers for a batch. `None` when there is nothing to apply.

    Returns `None` rather than a vector of 1.0s when the tracker is absent/disabled, so
    a caller composing this with another weight vector (via `LossWeighter`'s
    `extra_weights`) can tell "nothing to multiply in" from "multiply by exactly 1" --
    the two behave identically numerically, but only the former should be able to make
    an otherwise-unweighted run (`ivalue_loss_weight=none`) skip reweighting entirely.
    """
    import torch

    if tracker is None or not tracker.enabled or not nodes:
        return None
    weights = [tracker.multiplier(node, clip=clip) for node in nodes]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def pool_targeting_for(trainer):
    """Which object -- if any -- should fill `IValueTraversal`'s `group_targeting=` slot.

    `--ivalue-fairness-selection` swaps `trainer.fairness_tracker` in for
    `trainer.group_targeting`; both expose the same `is_targeted(node)` interface, so
    the traversal itself needs no changes to accept either. One function, called from
    both `test_hierarchical.create_traversal` and `AdaptiveTrainer._create_traversal`
    (the project's two IValueTraversal factories -- see either's docstring for why a
    decision like this has drifted between them before), so the choice cannot diverge.
    """
    if getattr(trainer, 'ivalue_fairness_selection', False):
        return getattr(trainer, 'fairness_tracker', None)
    return getattr(trainer, 'group_targeting', None)


__all__ = [
    "DEFAULT_TARGET_GROUPS", "GroupPerformanceTracker", "fairness_weights_for_batch",
    "pool_targeting_for",
]
