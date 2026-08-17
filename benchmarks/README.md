# Committed baselines

`baseline_<suite>.csv` and `baseline_<suite>.manifest.json` are written here by

    python development_tools/sweep.py promote <sweep-id>

and are meant to be **committed**, so the git history of this directory is the history of
the project's baselines and any checkout can be compared against one:

    python development_tools/sweep.py run --suite standard --compare-to baseline

Only the scored table lives here -- a few hundred kilobytes of tidy long rows. The record
tables behind it are hundreds of megabytes and stay under the gitignored `run_outputs/`, so
on a fresh checkout the per-sample paired test has nothing to align and the comparison
reports `-` for it rather than inventing one. Everything else still compares.

The manifest carries the git commit, the seed, the determinism mode, and the sha256 of the
node cache the baseline was measured against. A later comparison prints those first and
warns when any of them has moved on -- rebuilding the node cache is the one that matters,
because the two sweeps then scored different samples and the difference is not attributable
to a code change.

See [../docs/dev_sweep.md](../docs/dev_sweep.md).
