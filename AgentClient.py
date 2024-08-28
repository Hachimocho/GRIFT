


# Start up trainer
trainer = globals()[next(iter(wandb.config["trainer"]))](wandb.config["trainer"][next(iter(wandb.config["trainer"]))], wandb.config)
trainer.run()
trainer.test()  
