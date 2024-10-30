import os
from datasets.Dataset import Dataset
import pandas as pd
import numpy as np

class AIFaceDataset(Dataset):
    tags = ["deepfakes", "image"]
    hyperparameters = None

    def load(self):
        data_root = self.data_root
        train_csv = os.path.join(data_root, "train.csv")
        val_csv = os.path.join(data_root, "val.csv")
        test_csv = os.path.join(data_root, "test.csv")
        # If val_csv doesn't exist, randomly assign 
        if not os.path.exists(val_csv):
            try:
                df = pd.read_csv(train_csv)
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find input file: {train_csv}")
            except pd.errors.EmptyDataError:
                raise ValueError("The input file is empty")
                
            # Create random mask where ~70% of the data goes to training
            mask = np.random.random(len(df)) < .7
            
            # Split the dataframe
            df1 = df[mask]
            df2 = df[~mask]
            
            # Save to output files
            df1.to_csv(train_csv, index=False)
            df2.to_csv(val_csv, index=False)
            
        for subset, csv in zip(["train", "val", "test"], [train_csv, val_csv, test_csv]):
            for row in pd.read_csv(csv):
                image_path = os.path.join(data_root, row["Image Path"])
                self.nodes.append(subset, self.data_class(image_path, **self.data_args), [], int(row["Target"]), **self.node_args)

        return self.nodes