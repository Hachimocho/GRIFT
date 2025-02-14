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

    def _load_additional_attributes(self, subset):
        """Load additional attributes from CSV file for the given split"""
        csv_path = os.path.join(self.data_root, f"{subset}_attributes.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: No additional attributes file found at {csv_path}")
            return {}
        
        try:
            df = pd.read_csv(csv_path)
            # Create a mapping of filename to attributes
            attributes_map = {}
            for _, row in df.iterrows():
                filename = row['filename']
                # Convert row to dict, excluding the filename column
                attrs = row.drop('filename').to_dict()
                # Clean up the attributes
                attrs = {k: v for k, v in attrs.items() if pd.notna(v)}  # Remove NaN values
                attributes_map[filename] = attrs
            
            print(f"Loaded {len(attributes_map)} additional attributes for {subset}")
            return attributes_map
        except Exception as e:
            print(f"Warning: Error loading additional attributes for {subset}: {str(e)}")
            return {}

    def create_node(self, args):
        subset, path, target, gender, race, age, data_class, data_args, node_class, node_args, additional_attrs = args
        # Start with base attributes
        attributes = {
            "Gender": gender,
            "Race": race,
            "Age": age
        }
        
        # Add additional attributes if available
        if additional_attrs:
            filename = os.path.basename(path)
            if filename in additional_attrs:
                attributes.update(additional_attrs[filename])
        
        node = node_class(
            subset,
            data_class(path, **data_args),
            [],
            int(target),
            attributes,
            **node_args
        )
        # Explicitly set the split attribute
        node.split = subset
        return node

    def load(self):
        print("Load called.")
        data_root = self.data_root
        train_csv = os.path.join(data_root, "train.csv")
        val_csv = os.path.join(data_root, "val.csv")
        test_csv = os.path.join(data_root, "test.csv")
        
        # Check if we need to create splits
        if not os.path.exists(train_csv) or not os.path.exists(val_csv) or not os.path.exists(test_csv):
            try:
                # Load all data
                all_data = pd.read_csv(os.path.join(data_root, "data.csv"))
                
                # Create splits
                print("Creating train/val/test splits...")
                train_ratio, val_ratio = 0.7, 0.15  # This leaves 0.15 for test
                
                # Randomly shuffle the data
                all_data = all_data.sample(frac=1, random_state=42)
                
                # Calculate split indices
                n = len(all_data)
                train_end = int(train_ratio * n)
                val_end = int((train_ratio + val_ratio) * n)
                
                # Split the data
                train_data = all_data[:train_end]
                val_data = all_data[train_end:val_end]
                test_data = all_data[val_end:]
                
                # Save splits
                train_data.to_csv(train_csv, index=False)
                val_data.to_csv(val_csv, index=False)
                test_data.to_csv(test_csv, index=False)
                
                print(f"Created splits:")
                print(f"Train: {len(train_data)} samples")
                print(f"Val: {len(val_data)} samples")
                print(f"Test: {len(test_data)} samples")
                
                del all_data, train_data, val_data, test_data
                gc.collect()
            except Exception as e:
                raise Exception(f"Error creating splits: {str(e)}")
        
        # Store nodes for each subset separately
        all_nodes = []
        num_processes = min(4, cpu_count() - 1)
        
        # Load additional attributes for each split
        print("Loading additional attributes...")
        train_attrs = self._load_additional_attributes("train")
        val_attrs = self._load_additional_attributes("val")
        test_attrs = self._load_additional_attributes("test")
        
        # Process train, val, test in order
        for subset, csv, attrs in tqdm(list(zip(
            ["train", "val", "test"], 
            [train_csv, val_csv, test_csv],
            [train_attrs, val_attrs, test_attrs]
        )), desc="Loading datasets"):
            print(f"\nLoading {subset} data for AIFace dataset...")
            
            if not os.path.exists(csv):
                print(f"Warning: {csv} not found, skipping {subset} split")
                continue
                
            try:
                df = pd.read_csv(csv)
                if len(df) == 0:
                    print(f"Warning: {csv} is empty, skipping {subset} split")
                    continue
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find input file: {csv}")
            except pd.errors.EmptyDataError:
                raise ValueError("The input file is empty")
                
            chunk_size = 100000
            subset_nodes = []
            
            for chunk in pd.read_csv(csv, chunksize=chunk_size):
                chunk['image_path'] = chunk['Image Path'].apply(lambda x: os.path.join(data_root, x.lstrip('/')))
                args_list = [
                    (subset, path, target, gender, race, age, self.data_class, self.data_args, self.node_class, self.node_args, attrs)
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