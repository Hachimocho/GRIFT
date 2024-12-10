import os
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
import gc
from datasets.Dataset import Dataset

class CDFDataset(Dataset):
    tags = ["deepfakes", "image", "attributes"]
    hyperparameters = None
    
    def create_node(self, args):
        """Create a node with the given arguments"""
        subset, path, target, data_class, data_args, node_class, node_args = args
        try:
            # Initialize node with correct number of arguments
            return node_class(
                subset,
                data_class(path, **data_args),
                [],  # empty edges list
                target
            )
        except AssertionError:
            print(f"Warning: Could not load image: {path}")
            return None

    def _collect_image_paths(self, data_root):
        """Collect all image paths and their corresponding labels"""
        image_info = []
        
        # Create progress bar for class folders
        for class_folder in tqdm(["fake", "real"], desc="Processing class folders"):
            target = 1 if class_folder == "fake" else 0
            
            if class_folder == "fake":
                sub_folders = ["Celeb-synthesis"]
            else:
                sub_folders = ["Celeb-real", "YouTube-real"]
                
            # Progress bar for subfolders
            for sub_folder in tqdm(sub_folders, desc=f"Processing {class_folder} subfolders", leave=False):
                # Progress bar for splits
                for split in tqdm(["train", "val", "test"], desc=f"Processing {sub_folder} splits", leave=False):
                    folder_path = os.path.join(data_root, class_folder, sub_folder, split)
                    if not os.path.exists(folder_path):
                        print(f"Warning: Path does not exist: {folder_path}")
                        continue
                    
                    # Process all items in the folder with progress bar
                    items = os.listdir(folder_path)
                    for item in tqdm(items, desc=f"Processing {split} items", leave=False):
                        item_path = os.path.join(folder_path, item)
                        
                        # If directory, look for frames
                        if os.path.isdir(item_path):
                            frames = [f for f in os.listdir(item_path) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            for frame in tqdm(frames, desc=f"Processing frames in {item}", leave=False):
                                frame_path = os.path.join(item_path, frame)
                                image_info.append((split, frame_path, target))
                        # If image file, use directly
                        elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_info.append((split, item_path, target))
        
        return image_info

    def load(self):
        print("Loading CDF dataset...")
        data_root = self.data_root
        
        # Collect all image paths and their labels
        print("\nCollecting image paths...")
        image_info = self._collect_image_paths(data_root)
        
        if not image_info:
            raise ValueError(f"No valid image files found in {data_root}")
            
        print(f"\nFound {len(image_info)} images")
            
        # Group by split
        print("\nGrouping images by split...")
        split_groups = {}
        for split, path, target in tqdm(image_info, desc="Organizing splits"):
            if split not in split_groups:
                split_groups[split] = []
            split_groups[split].append((path, target))
            
        # Clear image_info to free memory
        del image_info
        gc.collect()
            
        # Store nodes for each subset separately
        all_nodes = []
        # Use more processes since we're I/O bound
        num_processes = min(6, cpu_count())
        
        # Progress bar for splits
        for split in tqdm(split_groups.keys(), desc="Processing splits"):
            items = split_groups[split]
            print(f"\nLoading {split} data ({len(items)} images)...")
            
            # Larger chunk size for better I/O efficiency
            chunk_size = 1000
            num_chunks = (len(items) - 1) // chunk_size + 1
            
            # Progress bar for chunks
            for chunk_idx in tqdm(range(num_chunks), desc=f"Processing {split} chunks"):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(items))
                chunk_items = items[start_idx:end_idx]
                
                args_list = [
                    (split, path, target, self.data_class, self.data_args, self.node_class, self.node_args)
                    for path, target in chunk_items
                ]
                
                # Process chunk with worker pool
                with Pool(num_processes) as pool:
                    chunk_nodes = list(tqdm(
                        pool.imap(self.create_node, args_list, chunksize=50),
                        total=len(args_list),
                        desc=f"Creating nodes for chunk {chunk_idx + 1}/{num_chunks}",
                        leave=False
                    ))
                    
                    # Filter out None values from failed loads
                    chunk_nodes = [node for node in chunk_nodes if node is not None]
                    all_nodes.extend(chunk_nodes)
                
                # Explicitly clean up memory after each chunk
                del args_list
                del chunk_nodes
                pool.close()
                pool.join()
                gc.collect()
        
        # Clean up split groups
        del split_groups
        gc.collect()
        
        self.nodes = all_nodes
        print(f"\nLoaded {len(self.nodes)} nodes from CDF dataset")
        return self.nodes