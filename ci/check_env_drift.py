#!/usr/bin/env python3
"""Report where environment.ci.yml diverges from environment.yml.

`environment.yml` is the training environment's source of truth: 340 build-pinned
packages against CUDA 11.8. `environment.ci.yml` is a small hand-written CPU
environment for the fast test tier. They are *supposed* to differ -- this script
exists so the divergence is visible rather than discovered when CI passes on a
version the training box does not have.

Runs on stdlib only, so it needs no environment of its own. Prints a report and
always exits 0; the CI job that calls it is informational.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Packages the CI env deliberately omits, with the reason.
EXPECTED_ABSENT = {
    "pytorch-cuda": "CPU runner",
    "pytorch-mutex": "CPU runner",
    "torchtriton": "CPU runner",
    "wandb": "no experiment logging in CI",
    "flask": "the web UI is not exercised by the fast tier",
}

#: Packages the CI env adds that the training env lacks.
EXPECTED_EXTRA = {
    "pytest": "not in environment.yml until this work added it",
    "pytest-cov": "coverage reporting",
    "pytest-timeout": "per-test timeouts",
    "python-louvain": "absent from the training env, where Louvain is a silent no-op",
    "matplotlib-base": "headless replacement for matplotlib",
}


def parse_packages(path):
    """Extract bare package names from a conda environment file.

    A deliberately loose parser: yaml is not guaranteed to be importable in the
    stdlib-only context this runs in.
    """
    names = {}
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ") or stripped.startswith("- pip"):
            continue
        spec = stripped[2:].strip()
        if not spec or spec.startswith("#"):
            continue
        name = re.split(r"[=<>!\s]", spec, maxsplit=1)[0].strip()
        if name:
            names[name.lower()] = spec
    return names


def main():
    training_path = REPO_ROOT / "environment.yml"
    ci_path = REPO_ROOT / "environment.ci.yml"
    if not training_path.exists() or not ci_path.exists():
        print(f"missing environment file: {training_path} or {ci_path}")
        return 0

    training = parse_packages(training_path)
    ci = parse_packages(ci_path)

    print(f"environment.yml:    {len(training)} packages")
    print(f"environment.ci.yml: {len(ci)} packages")

    unexplained_extra = []
    print("\n-- in CI but not in the training env --")
    for name in sorted(set(ci) - set(training)):
        reason = EXPECTED_EXTRA.get(name)
        print(f"  {name:24s} {reason or '** UNEXPLAINED **'}")
        if reason is None:
            unexplained_extra.append(name)

    print("\n-- version skew on shared packages --")
    skewed = 0
    for name in sorted(set(ci) & set(training)):
        training_spec, ci_spec = training[name], ci[name]
        # The CI env is mostly unpinned, so only report where CI pins something
        # different from what the training env pins.
        if "=" in ci_spec and ci_spec.split("=")[1:] != training_spec.split("=")[1:]:
            print(f"  {name:24s} training={training_spec}  ci={ci_spec}")
            skewed += 1
    if not skewed:
        print("  (none: the CI env leaves shared packages unpinned)")

    if unexplained_extra:
        print(
            f"\nNOTE: {len(unexplained_extra)} CI package(s) have no recorded reason: "
            f"{unexplained_extra}. Add them to EXPECTED_EXTRA in this script, or "
            f"remove them from environment.ci.yml."
        )
    else:
        print("\nAll CI-only packages have a recorded reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
