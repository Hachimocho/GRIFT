"""Target demographic *groups* by mean I-value, then sample uniformly inside them.

The motivation is a specific failure. Instance-level selection by predicted I-value has lost
every comparison run so far, and the leading reasons are both about individual samples: on a
corpus that is 87.55% one class and carries label noise, the highest-scoring individual
samples are disproportionately outliers and mislabels, and Phase 0 measured that picking them
by walking a similarity graph produces batches 2.3x less diverse than i.i.d. ones.

A group mean is immune to both. Averaging I-value over every node sharing a
race/gender/age combination cancels per-sample noise, and once a group is chosen, sampling
*uniformly within it* keeps the within-batch diversity that instance-level selection destroys.
So this asks a different question from every earlier arm: not "which sample is most
informative" but "which *kind* of face is the model weakest on".

It also connects the method to what the project measures anyway -- `evaluation/uq/subgroups.py`
already reports per-subgroup accuracy, calibration and worst-group disparity, so an arm that
deliberately targets weak groups can be read against those numbers directly.

Group means are accumulated from `AdaptiveTrainer.get_i_value`, the single funnel every
predicted I-value already passes through -- the same hook `PerformanceGraphManager` uses -- so
this costs no extra DQN forward passes.
"""

#: Attribute keys defining a group, tried in order per axis. These are the names as they exist
#: on a *node*, which are not the `gt_*` names the record tables use.
GROUP_ATTRIBUTES = (
    ("Ground Truth Race", "gt_race", "race"),
    ("Ground Truth Gender", "gt_gender", "gender"),
    ("Ground Truth Age", "gt_age", "age"),
)

#: A group needs this many observations before its mean is trusted enough to target. Without
#: it the first few nodes seen would decide which groups get the whole run's attention.
MIN_GROUP_OBSERVATIONS = 50

#: How many groups are targeted by default.
DEFAULT_TOP_GROUPS = 3


def group_key(node):
    """The group a node belongs to, or None when its attributes are missing."""
    attributes = getattr(node, "attributes", None) or {}
    parts = []
    for candidates in GROUP_ATTRIBUTES:
        for key in candidates:
            if key in attributes and attributes[key] is not None:
                parts.append(str(attributes[key]))
                break
        else:
            return None
    return "|".join(parts)


class GroupTargeting:
    """Running mean I-value per group, and the set currently targeted."""

    def __init__(self, top_groups=DEFAULT_TOP_GROUPS,
                 min_observations=MIN_GROUP_OBSERVATIONS, enabled=False):
        self.enabled = bool(enabled)
        self.top_groups = max(1, int(top_groups))
        self.min_observations = max(1, int(min_observations))
        self._means = {}       # group -> (mean_i_value, count)
        self._targeted = ()    # cached, recomputed when asked

    def observe(self, node, i_value):
        """Fold one predicted I-value into its group's mean. O(1)."""
        if not self.enabled:
            return
        key = group_key(node)
        if key is None:
            return
        try:
            value = float(i_value)
        except (TypeError, ValueError):
            return
        mean, count = self._means.get(key, (0.0, 0))
        self._means[key] = (mean + (value - mean) / (count + 1), count + 1)
        self._targeted = ()

    def targeted_groups(self):
        """The `top_groups` groups with the highest mean I-value, among those seen enough."""
        if self._targeted:
            return self._targeted
        eligible = [
            (key, mean) for key, (mean, count) in self._means.items()
            if count >= self.min_observations
        ]
        if not eligible:
            return ()
        eligible.sort(key=lambda item: (-item[1], item[0]))
        self._targeted = tuple(key for key, _mean in eligible[:self.top_groups])
        return self._targeted

    def is_targeted(self, node):
        """True when `node` is in a targeted group.

        True while no group qualifies yet, so an early epoch samples normally instead of
        rejecting everything and starving the traversal.
        """
        if not self.enabled:
            return True
        targeted = self.targeted_groups()
        if not targeted:
            return True
        key = group_key(node)
        return key is None or key in targeted

    def summary(self):
        targeted = self.targeted_groups()
        return {
            "groups_seen": len(self._means),
            "targeted": list(targeted),
            "targeted_means": [round(self._means[k][0], 4) for k in targeted],
        }


__all__ = [
    "DEFAULT_TOP_GROUPS", "GROUP_ATTRIBUTES", "GroupTargeting",
    "MIN_GROUP_OBSERVATIONS", "group_key",
]
