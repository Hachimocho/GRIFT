import os
from datasets.Dataset import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc

class AIFaceDataset(Dataset):
    tags = ["deepfakes", "image", "attributes"]
    hyperparameters = None

    def create_node(self, args):
        subset, path, target, gender, race, age, data_class, data_args, node_class, node_args = args
        return node_class(
            subset,
            data_class(path, **data_args),
            [],
            int(target),
            {"Gender": gender, "Race": race, "Age": age},
            **node_args
        )

    def load(self):
        print("Load called.")
        data_root = self.data_root
        train_csv = os.path.join(data_root, "train.csv")
        val_csv = os.path.join(data_root, "val.csv")
        test_csv = os.path.join(data_root, "test.csv")
        
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"Could not find input file: {train_csv}")
        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"Could not find input file: {test_csv}")
        if not os.path.exists(val_csv) or os.stat(val_csv).st_size == 0:
            try:
                df = pd.read_csv(train_csv)
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find input file: {train_csv}")
            except pd.errors.EmptyDataError:
                raise ValueError("The input file is empty")
                
            print("AIFace loaded for first time, generating train/val splits...")
            mask = np.random.random(len(df)) < .7
            df1 = df[mask]
            df2 = df[~mask]
            df1.to_csv(train_csv, index=False)
            df2.to_csv(val_csv, index=False)
            del df, df1, df2
            gc.collect()
            
        # Store nodes for each subset separately
        all_nodes = []
        num_processes = min(4, cpu_count() - 1)
        
        for subset, csv in tqdm(list(zip(["val", "train", "test"], [val_csv, train_csv, test_csv])), desc="Loading datasets"):
            print(f"\nLoading {subset} data for AIFace dataset...")
            
            chunk_size = 100000
            subset_nodes = []
            
            for chunk in pd.read_csv(csv, chunksize=chunk_size):
                chunk['image_path'] = chunk['Image Path'].apply(lambda x: os.path.join(data_root, x.lstrip('/')))
                args_list = [
                    (subset, path, target, gender, race, age, self.data_class, self.data_args, self.node_class, self.node_args)
                    for path, target, gender, race, age in zip(
                        chunk['image_path'],
                        chunk['Target'],
                        chunk['Ground Truth Gender'],
                        chunk['Ground Truth Race'],
                        chunk['Ground Truth Age']
                    )
                ]
                
                with Pool(num_processes) as pool:
                    chunk_nodes = list(tqdm(
                        pool.imap_unordered(self.create_node, args_list, chunksize=1000),
                        total=len(args_list),
                        desc=f"Processing {subset} chunk"
                    ))
                    subset_nodes.extend(chunk_nodes)
                
                del chunk
                del args_list
                del chunk_nodes
                gc.collect()
            
            # Add all nodes from this subset to the final list
            all_nodes.extend(subset_nodes)
            del subset_nodes
            gc.collect()
            
        self.nodes = all_nodes
        return self.nodes