#!/usr/bin/env python3
"""Suite definitions and matrix expansion for the development sweep.

A full cross product of the usable detectors, traversals, graph types, uncertainty heads,
and graph updaters is several hundred training runs. That is the wrong shape for a
development check: nearly every cell answers the same question twice, and the sweep
becomes something you run once and never again.

So a suite is a **reference cell plus per-axis variants**. Varying one axis at a time
exercises every implementation of every component in ``1 + sum(len(axis) - 1)`` runs
instead of their product -- sixteen runs rather than four hundred and eighty for the
standard suite. That is enough to answer "did my change break any traversal, detector,
head, or updater?", which is the question. Interactions are opt-in through ``--cross``,
because an interaction is a research question rather than a regression check.

Expansion prunes cells that cannot produce a result, always with a reason:

* Detectors the capability table marks ``BROKEN`` (`validate_architectures`).
* Updaters that need predicted I-values paired with a traversal that supplies none --
  ``GraphReductionManager`` raises on a trainer with no ``get_i_value``, and
  ``PerformanceGraphManager`` reads a neutral default for every node and rewires nothing.
* Subcluster traversals when ``python-louvain`` is absent, because
  ``HyperGraph.assign_louvain_subclusters`` is then a documented no-op and the cell would
  quietly run its non-subcluster fallback while claiming to test subclustering.

A pruned cell is recorded and reported, never silently dropped: a matrix with explained
holes is honest, a matrix with missing rows is not.
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Settings every suite forces on, whatever the reference cell says.
#:
#: `uq_records` is the whole point: the stdout metrics dict carries batch means of raw
#: uncertainty signals on incomparable scales, which cannot answer whether anything got
#: better. Every comparison number comes from the per-sample tables instead.
#:
#: `determinism: strict` is what makes a *paired* comparison legitimate. Same seed, same
#: node cache, strict mode: baseline and candidate score the same samples in the same
#: order, so a delta is attributable to the code change rather than to GPU
#: nondeterminism, which `docs/testing.md` measures at about 3e-2 in probability space.
FORCED = {
    "uq_records": True,
    # val too: temperature scaling must be fitted on data the test numbers never saw.
    "uq_records_splits": "val,test",
    # Graph-distance uncertainty needs test-split edges, and the sweep scores test.
    "build_val_test_edges": True,
    # Keeps the legacy bias_metrics block in the logs commensurable with the subgroup
    # numbers the report computes from records.
    "enable_val_bias_inference": True,
    "determinism": "strict",
    "export_csv_per_run": False,
    # Threshold-free, so it cannot freeze the best epoch at 1 the way accuracy does on an
    # imbalanced split -- and while it is frozen, no axis acting after epoch 1 (either graph
    # updater, later training) can be measured at all.
    "checkpoint_metric": "auroc",
    # Fitted on val, applied to test, reported alongside the 0.5 numbers. Costs nothing:
    # it reads the record tables the run already wrote.
    "tune_threshold": True,
    "threshold_objective": "balanced_accuracy",
}

#: Reference cell shared by the real-data suites. Deliberately the cheapest configuration
#: that still produces every column the report wants: `resnestdf` is SUPPORTED and small,
#: `random` needs no DQN warm-up, `nonclustered` skips the clustering build.
_REFERENCE = {
    "architectures": ["resnestdf"],
    "traversal_type": "random",
    "graph_type": "nonclustered",
    "uncertainty_head": "none",
    "graph_manager": "none",
    "reduction_enabled": False,
    "seed": 42,
    "batch_size": 32,
    "cached_nodes": True,
    "cache_file": "node_cache/cached_nodes.pkl",
    "num_workers": 2,
    "val_num_workers": 2,
}

#: Axis name -> the config keys each variant sets. A variant is a dict merged over the
#: reference cell, so an axis can move more than one key at once -- which the updater axis
#: needs, since a reduction strategy is a strategy plus a percentage.
AXES = {
    "arch": {
        "effnetdf": {"architectures": ["effnetdf"]},
        "vistransformdf": {"architectures": ["vistransformdf"]},
        "swintransformdf": {"architectures": ["swintransformdf"]},
        "squeezenetdf": {"architectures": ["squeezenetdf"]},
    },
    "traversal": {
        "comprehensive": {"traversal_type": "comprehensive"},
        # One I-value traversal, twice -- the walk it uses is a property of the graph, so
        # the pairing with --graph-type is what distinguishes these two cells.
        "i-value": {"traversal_type": "i-value", "graph_type": "nonclustered"},
        "i-value-clustered": {"traversal_type": "i-value", "graph_type": "clustered"},
    },
    "graph": {
        # The graph-*construction* contrast: edges built within race-gender groups versus
        # across all nodes. The reference cell is `nonclustered`, so this is the other half,
        # and `RandomTraversal` walks adjacency, so the difference does reach training.
        #
        # The two `*_subclustered` types are deliberately absent. Subclustering assigns node
        # attributes and changes no edges, so only a traversal that consults
        # `graph.subclusters` can respond to it -- and those two traversals are on the
        # traversal axis already, paired with the graph type they need. Listed here against
        # the reference's `random` traversal they produced record tables bit-identical to
        # `clustered`: two cells, forty minutes, measuring nothing. `axis_constraints` now
        # refuses that pairing outright.
        "clustered": {"graph_type": "clustered"},
    },
    "head": {
        "evidential": {"uncertainty_head": "evidential"},
        "batchensemble": {"uncertainty_head": "batchensemble"},
        "sngp": {"uncertainty_head": "sngp"},
        # Not a head, but the same axis in practice: an evaluation-time sampling method
        # selected by a count rather than a head name.
        "mc_dropout": {"uncertainty_head": "none", "mc_dropout_samples": 8,
                       "uncertainty_dropout_rate": 0.2},
    },
    "updater": {
        # Rewiring reads predicted I-values, so it is paired with a traversal that
        # produces them. Without that pairing every node sits at the neutral default and
        # the manager rewires nothing -- see axis_constraints.
        # Prunes already-learned nodes. Paired with an i-value traversal because the
        # quantiles are computed over predicted I-values, and without a DQN those are random
        # draws -- see axis_constraints.
        "performance": {
            "graph_manager": "performance", "traversal_type": "i-value",
            "graph_updates_per_epoch": 4, "removal_fraction": 0.02,
            "graph_remove_target": "strong",
        },
        "performance_prune_weak": {
            "graph_manager": "performance", "traversal_type": "i-value",
            "graph_updates_per_epoch": 4, "removal_fraction": 0.02,
            "graph_remove_target": "weak",
        },
        "reduce_max_ival": {
            "reduction_enabled": True, "reduction_strategy": "max_ival",
            "reduction_percentage": 10.0, "reduction_interval": "end_of_epoch",
            "traversal_type": "i-value",
        },
        "reduce_random": {
            "reduction_enabled": True, "reduction_strategy": "random",
            "reduction_percentage": 10.0, "reduction_interval": "end_of_epoch",
        },
        "reduce_restore": {
            "reduction_enabled": True, "reduction_strategy": "random",
            "reduction_percentage": 10.0, "restoration_strategy": "random_pool",
            "restoration_percentage": 50.0,
        },
    },
}

#: Traversals that supply predicted I-values, and therefore satisfy the updaters that
#: need them. The subcluster variants are included because they subclass the I-value
#: traversals -- `CapabilityManager.configure_for_traversal`'s allowlist omits them, which
#: is a separate drift worth knowing about but does not change what they compute.
IVALUE_TRAVERSALS = frozenset({"i-value"})

#: Reduction strategies that read `trainer.get_i_value`.
IVALUE_REDUCTIONS = frozenset({"max_ival", "min_ival", "mix_max_ival"})

#: Traversals that consult `graph.subclusters`. Empty since the two subcluster traversals
#: were removed: nothing reads subcluster assignments any more, so no traversal can respond
#: to a `*_subclustered` graph type. Kept as a named constant rather than inlined, because
#: `axis_constraints` still needs to refuse that pairing and a future traversal could
#: repopulate it.
SUBCLUSTER_TRAVERSALS = frozenset()

SUITES = {
    "smoke": {
        "description": "Cheapest possible pass: does anything run end to end?",
        "reference": {
            **_REFERENCE,
            "num_epochs": 1,
            "cached_nodes_count": 200,
            "train_steps": 50,
            "val_steps": 50,
        },
        "axes": ["traversal"],
        # One variant per axis keeps this a handful of minutes.
        "axis_limit": 2,
    },
    "standard": {
        "description": "One variant per implementation of every component.",
        "reference": {
            **_REFERENCE,
            "num_epochs": 3,
            "cached_nodes_count": 2000,
            "train_steps": 500,
            "val_steps": 500,
        },
        "axes": ["arch", "traversal", "graph", "head", "updater"],
    },
    "full": {
        "description": "Standard, at realistic epoch counts and the full node cache.",
        "reference": {
            **_REFERENCE,
            "num_epochs": 10,
            "cached_nodes_count": 5000,
            "train_steps": 1000,
            "val_steps": 1000,
        },
        "axes": ["arch", "traversal", "graph", "head", "updater"],
        # Detector x traversal is the interaction most likely to hide a real bug: a
        # traversal that only misbehaves on one backbone's feature space.
        "cross": ["arch", "traversal"],
    },
}


class SuiteError(ValueError):
    """Raised when a suite definition or a filter cannot be resolved."""


class Cell:
    """One planned run: an identity, a config, and possibly a reason it will not run."""

    __slots__ = ("cell_id", "axis", "axis_value", "config", "skip_reason")

    def __init__(self, cell_id, axis, axis_value, config, skip_reason=None):
        self.cell_id = cell_id
        self.axis = axis
        self.axis_value = axis_value
        self.config = config
        self.skip_reason = skip_reason

    @property
    def runnable(self):
        return self.skip_reason is None

    @property
    def detector(self):
        architectures = self.config.get("architectures") or []
        return architectures[0] if architectures else "unknown"

    @property
    def description(self):
        """The output subdirectory `test_hierarchical.py` will use for this cell.

        Mirrors how `test_configs` builds `description`, which is what names the
        per-configuration directory under `run_outputs/<run_id>/` and keys the results
        block in `determinism.json`. Recomputing it here rather than parsing it back out
        means the sweep knows where a cell's records will land before it launches.
        """
        return f"{self.detector}_{self.config.get('traversal_type')}"

    def to_dict(self):
        return {
            "cell_id": self.cell_id,
            "axis": self.axis,
            "axis_value": self.axis_value,
            "detector": self.detector,
            "description": self.description,
            "config": self.config,
            "skip_reason": self.skip_reason,
        }

    def __repr__(self):
        state = "skip" if self.skip_reason else "run"
        return f"<Cell {self.cell_id} [{state}]>"


def load_suite(name, suite_file=None):
    """Resolve a suite by name, optionally overridden by a JSON file.

    A suite file may hold either a full ``{name: suite}`` mapping or a single suite dict.
    Its keys merge over the built-in of the same name rather than replacing it, so a file
    can change ``num_epochs`` without restating every axis.
    """
    suites = dict(SUITES)
    if suite_file:
        with open(suite_file) as handle:
            loaded = json.load(handle)
        if "reference" in loaded or "axes" in loaded:
            loaded = {name: loaded}
        for key, override in loaded.items():
            merged = copy.deepcopy(suites.get(key, {}))
            reference = {**merged.get("reference", {}), **override.get("reference", {})}
            merged.update(override)
            if reference:
                merged["reference"] = reference
            suites[key] = merged

    if name not in suites:
        raise SuiteError(
            f"unknown suite {name!r}; available: {', '.join(sorted(suites))}"
        )
    suite = copy.deepcopy(suites[name])
    if "reference" not in suite:
        raise SuiteError(f"suite {name!r} has no 'reference' cell")
    return suite


def resolve_axes(suite, axes=None):
    """`{axis: {value: overrides}}` for the axes this suite uses.

    An axis named by a suite but absent from `AXES` is an error, not an empty axis: a
    typo would otherwise silently reduce coverage.
    """
    names = list(axes if axes is not None else suite.get("axes", []))
    unknown = [name for name in names if name not in AXES and name not in suite.get("extra_axes", {})]
    if unknown:
        raise SuiteError(
            f"unknown axis/axes {', '.join(unknown)}; available: {', '.join(sorted(AXES))}"
        )
    extra = suite.get("extra_axes", {})
    resolved = {}
    limit = suite.get("axis_limit")
    for name in names:
        variants = dict(extra.get(name) or AXES.get(name, {}))
        if limit:
            variants = dict(list(variants.items())[:limit])
        resolved[name] = variants
    return resolved


def expand(suite, axes=None, cross=None, only=None, allow_broken=False):
    """Build the cell list for a suite. Returns `[Cell]`, reference cell first.

    `cross` names axes to expand factorially against each other instead of one at a time.
    `only` filters to `axis=value` selectors -- `["traversal=i-value", "arch"]` keeps the
    named variant, or every variant of a bare axis name. The reference cell is always
    kept: without it there is nothing for the variants to be a variant of.
    """
    resolved = resolve_axes(suite, axes)
    # FORCED last, so a suite cannot accidentally turn record writing off -- except for
    # `forced_overrides`, which is how an explicit CLI flag overrides a forced default
    # (`--determinism fast`). Stated as data rather than a special case in the caller, so
    # the resulting config is still exactly what gets launched.
    reference = {
        **copy.deepcopy(suite["reference"]),
        **FORCED,
        **copy.deepcopy(suite.get("forced_overrides") or {}),
    }
    cross = [name for name in (cross or suite.get("cross") or []) if name in resolved]

    cells = [Cell("reference", "reference", "reference", reference)]

    if cross:
        cells += _expand_cross(reference, resolved, cross)
        remaining = {
            name: variants for name, variants in resolved.items() if name not in cross
        }
    else:
        remaining = resolved

    for axis, variants in remaining.items():
        for value, overrides in variants.items():
            config = _merge(reference, overrides)
            cells.append(Cell(f"{axis}={value}", axis, value, config))

    if only:
        cells = _filter(cells, only)

    return [_gate(cell, allow_broken=allow_broken) for cell in cells]


def summarize(cells):
    """Counts for a one-line plan summary."""
    runnable = [cell for cell in cells if cell.runnable]
    return {
        "total": len(cells),
        "runnable": len(runnable),
        "skipped": len(cells) - len(runnable),
        "detectors": sorted({cell.detector for cell in runnable}),
    }


def format_plan(cells):
    """A readable table of the planned matrix, skips included with their reasons."""
    lines = []
    width = max([len(cell.cell_id) for cell in cells] + [7])
    header = f"{'cell'.ljust(width)}  {'detector':<16} {'traversal':<32} {'graph':<28} head"
    lines.append(header)
    lines.append("-" * len(header))
    for cell in cells:
        config = cell.config
        marker = "" if cell.runnable else "  SKIP"
        lines.append(
            f"{cell.cell_id.ljust(width)}  {cell.detector:<16} "
            f"{str(config.get('traversal_type')):<32} "
            f"{str(config.get('graph_type')):<28} "
            f"{str(config.get('uncertainty_head'))}{marker}"
        )
        if not cell.runnable:
            lines.append(f"{' ' * width}    reason: {cell.skip_reason}")

    counts = summarize(cells)
    lines.append("")
    lines.append(
        f"{counts['runnable']} runnable, {counts['skipped']} skipped, "
        f"{counts['total']} total"
    )
    return "\n".join(lines)


# -- internals ------------------------------------------------------------------ #


def _merge(reference, overrides):
    config = copy.deepcopy(reference)
    config.update(copy.deepcopy(overrides))
    return config


def _expand_cross(reference, resolved, cross):
    """Factorial expansion of the named axes, excluding the reference combination."""
    import itertools

    grids = [list(resolved[name].items()) for name in cross]
    cells = []
    for combination in itertools.product(*grids):
        config = copy.deepcopy(reference)
        parts = []
        for axis, (value, overrides) in zip(cross, combination):
            config.update(copy.deepcopy(overrides))
            parts.append(f"{axis}={value}")
        cells.append(Cell("+".join(parts), "+".join(cross), "+".join(
            value for _axis, (value, _overrides) in zip(cross, combination)
        ), config))
    return cells


def _filter(cells, only):
    """Keep the reference plus cells matching any `axis` or `axis=value` selector."""
    selectors = []
    for entry in only:
        for part in str(entry).split(","):
            part = part.strip()
            if part:
                selectors.append(part)
    if not selectors:
        return cells

    kept = []
    matched = set()
    for cell in cells:
        if cell.axis == "reference":
            kept.append(cell)
            continue
        for selector in selectors:
            if selector == cell.axis or selector == cell.cell_id:
                kept.append(cell)
                matched.add(selector)
                break
            if "=" in selector:
                axis, _, value = selector.partition("=")
                if axis == cell.axis and value == str(cell.axis_value):
                    kept.append(cell)
                    matched.add(selector)
                    break

    unmatched = [
        selector for selector in selectors
        if selector not in matched and "+" not in selector
    ]
    if unmatched:
        raise SuiteError(
            f"--only selector(s) matched nothing: {', '.join(unmatched)}. "
            f"Use an axis name, or axis=value."
        )
    return kept


def _gate(cell, allow_broken=False):
    """Attach a skip reason when a cell cannot produce a meaningful result."""
    if cell.skip_reason:
        return cell
    reason = _architecture_problem(cell, allow_broken) or axis_constraints(cell.config)
    if reason:
        cell.skip_reason = reason
    return cell


def _architecture_problem(cell, allow_broken):
    from models.uncertainty.capabilities import validate_architectures

    names = list(cell.config.get("architectures") or [])
    if not names:
        return "no architecture configured"
    _usable, problems = validate_architectures(names, allow_broken=allow_broken)
    if problems:
        return "; ".join(f"{name}: {reason}" for name, reason in sorted(problems.items()))
    return None


def axis_constraints(config):
    """Why this config cannot produce a result, or None.

    These are training-configuration constraints, which is why they live here rather than
    in `evaluation/uq/registry.gate`: that gate reasons about what a *model* can produce,
    and none of these is about the model.
    """
    traversal = str(config.get("traversal_type", ""))
    has_ivalues = traversal in IVALUE_TRAVERSALS

    if config.get("graph_manager") == "performance" and not has_ivalues:
        return (
            f"graph_manager=performance rewires by predicted I-value, and traversal "
            f"{traversal!r} enables no DQN capability. Every node would read the neutral "
            f"default, so nothing would be rewired and the cell would be "
            f"indistinguishable from a static graph. Pair it with an i-value traversal."
        )

    strategy = str(config.get("reduction_strategy", "none"))
    if config.get("reduction_enabled") and strategy in IVALUE_REDUCTIONS and not has_ivalues:
        return (
            f"reduction_strategy={strategy} reads trainer.get_i_value, which traversal "
            f"{traversal!r} does not provide -- GraphReductionManager raises on it. Use "
            f"reduction_strategy=random, or pair with an i-value traversal."
        )

    if config.get("reduction_enabled") and strategy == "none":
        return "reduction_enabled with reduction_strategy=none removes nothing"

    if config.get("reduction_enabled") and not float(config.get("reduction_percentage", 0.0)):
        return "reduction_enabled with reduction_percentage=0 removes nothing"

    graph_type = str(config.get("graph_type", ""))
    subclustered_graph = graph_type.endswith("_subclustered")
    subcluster_traversal = traversal in SUBCLUSTER_TRAVERSALS

    # Both halves of the pairing have to agree, or the cell is bit-identical to a cheaper
    # one. This is not hypothetical: three `graph` cells came back with the same record
    # digest because subclustering cannot reach a traversal that never reads it.
    if subclustered_graph and not subcluster_traversal:
        pairing = (
            f"Pair it with one of: {', '.join(sorted(SUBCLUSTER_TRAVERSALS))}."
            if SUBCLUSTER_TRAVERSALS else
            "No traversal reads subcluster assignments any more -- the two that did were "
            "removed -- so every *_subclustered graph type is currently inert."
        )
        return (
            f"graph_type={graph_type} assigns Louvain subclusters, which are node "
            f"attributes and leave edges unchanged, so traversal {traversal!r} cannot "
            f"respond to them -- this cell would be identical to "
            f"{graph_type.replace('_subclustered', '')!r}. {pairing}"
        )
    if subcluster_traversal and not subclustered_graph:
        return (
            f"traversal {traversal!r} selects among subclusters, but graph_type="
            f"{graph_type} never assigns them, so it silently falls back to its "
            f"no-subcluster path. Use a *_subclustered graph type."
        )

    wants_subclusters = subclustered_graph or subcluster_traversal
    if wants_subclusters and not _louvain_available():
        return (
            "subclustering needs python-louvain, which is not installed; "
            "HyperGraph.assign_louvain_subclusters is then a no-op, so this cell would "
            "silently run its non-subcluster fallback while claiming to test "
            "subclustering. Install python-louvain to enable it."
        )

    head = str(config.get("uncertainty_head", "none"))
    if head == "none" and int(config.get("mc_dropout_samples", 0) or 0) > 0:
        # Not a skip: MC dropout works off the base head's dropout sites. Recorded here
        # only so the reader knows it was considered.
        return None

    return None


def _louvain_available():
    """Whether `HyperGraph.assign_louvain_subclusters` will actually do anything."""
    import importlib.util

    return importlib.util.find_spec("community") is not None


__all__ = [
    "AXES", "Cell", "FORCED", "IVALUE_REDUCTIONS", "IVALUE_TRAVERSALS", "SUITES",
    "SuiteError", "axis_constraints", "expand", "format_plan", "load_suite",
    "resolve_axes", "summarize",
]
