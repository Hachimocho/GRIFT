from datasets import *
from dataloaders import *
from managers import *
from trainers.Trainer import Trainer

class DeepfakeTrainer(Trainer):
    """
    Trainer class for deepfakes.
    """
    tags = ["deepfakes"]
    hyperparameters = None

    def process_node_data(self):
        raise NotImplementedError("Overwrite this!")
    
    def train(self):
        for epoch in range(self.epochs):
            # Run until epoch ends
            while True:
                finished = False
                for model in self.models:
                    try:
                        self.train_traversal.traverse()
                    except RuntimeError as e:
                        # Epoch finished
                        finished = True
                        break
                if finished:
                    break
        
    def val(self):
        raise NotImplementedError("Overwrite this!")
    
    def test(self):
        raise NotImplementedError("Overwrite this!")
    
# try:
#     key_attributes = CHOSEN_ATTRIBUTES[wandb.config["key_attributes"]]
# except Exception as _:
#     print("Invalid key_attributes selection.")
#     sys.exit()
# try:
#     assert (wandb.config["num_models"] % (len(key_attributes)) * 2) == 0
    
# except Exception as _:
#     print("Invalid number of models for the selected key attributes. Number of models must be divsible by number of key attributes * 2.")
#     sys.exit()

# dataset = DeepfakeDataset(dataset_root='/home/brg2890/major/preprocessed', attribute_root='/home/brg2890/major/bryce_python_workspace/GraphWork/DeepEARL/attributes', splits_root = "/home/brg2890/major/bryce_python_workspace/GraphWork/DeepEARL/datasets", datasets=wandb.config["datasets"], auto_threshold=False, string_threshold=38)
# model = CNNModel(wandb.config["model"], wandb.config["frames_per_video"], dataset, wandb.config["warp_threshold"], wandb.config["num_models"], wandb.config["steps_per_epoch"], wandb.config["timesteps_before_return_allowed"], wandb.config["train_traversal_method"], wandb.config["hops_to_analyze"], wandb.config["val_test_traversal_method"], key_attributes, ATTRIBUTE_DICT)
# for epoch in tqdm(range(wandb.config["epochs"])):
#     model.train()
#     model.validate()
# model.test()