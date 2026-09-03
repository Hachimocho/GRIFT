# `import wandb` used to sit at module level, which made it a hard dependency of
# every caller of `utils` -- `utils/__init__.py` auto-imports every module in the
# package, so `import utils.visualize` (no wandb involved) transitively required
# wandb too. environment.ci.yml deliberately omits wandb (no experiment logging in
# CI, see ci/check_env_drift.py's EXPECTED_ABSENT), so that broke collection of
# every test that touched `dataloaders` or `utils`. Deferred into each function
# instead: these are only called from actual W&B logging code paths.

def save_tag(run, tag):
    import wandb

    # Save the tag to W&B
    artifact = wandb.Artifact("tag", type="string")
    artifact.add(wandb.Table(columns=["main"], data=[[tag]]), "tag_holder")

    run.log_artifact(artifact)

def load_tag(run):
    # Query W&B for an artifact and mark it as input to this run
    artifact = run.use_artifact("tag:latest")

    # Download the artifact's contents
    tag = artifact.get("tag_holder").get_column("main")[0]
    return tag

def load_tag_runless(project):
    import wandb

    # Query W&B for an artifact without an existing run
    api = wandb.Api()
    artifact = api.artifact(project + "/tag:latest")
    tag = artifact.get("tag_holder").get_column("main")[0]
    return tag