import wandb

wandb.login()
api = wandb.Api()
print(api._sweeps)