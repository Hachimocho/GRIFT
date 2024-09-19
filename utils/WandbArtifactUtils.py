import wandb

def save_tag(run, tag):
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
    # Query W&B for an artifact without an existing run
    api = wandb.Api()
    artifact = api.artifact(project + "/tag:latest")
    tag = artifact.get("tag_holder").get_column("main")[0]
    return tag