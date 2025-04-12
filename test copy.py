import sys
import traceback
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from trainers.ExperimentTrainer import ExperimentTrainer
from trainers.IValueTrainer import IValueTrainer
from dataloaders.UnclusteredDeepfakeDataloader import UnclusteredDeepfakeDataloader
from datasets.AIFaceDataset import AIFaceDataset
from data.ImageFileData import ImageFileData
from nodes.atrnode import AttributeNode
from managers.NoGraphManager import NoGraphManager
from managers.PerformanceGraphManager import PerformanceGraphManager
from traversals.ComprehensiveTraversal import ComprehensiveTraversal
from traversals.IValueTraversal import IValueTraversal
from traversals.RandomTraversal import RandomTraversal
from models.CNNModel import CNNModel
from edges.Edge import Edge
import copy
import torch
import time
import random

@contextmanager
def capture_output(filename):
    """Capture all stdout and stderr output to a file while still printing to terminal"""
    class TeeStream:
        def __init__(self, stdout, logfile):
            self.stdout = stdout
            self.logfile = logfile
            
        def write(self, message):
            self.stdout.write(message)
            self.logfile.write(message)
            
        def flush(self):
            self.stdout.flush()
            self.logfile.flush()
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logpath = log_dir / filename
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        with open(logpath, 'w') as logfile:
            tee_stdout = TeeStream(old_stdout, logfile)
            tee_stderr = TeeStream(old_stderr, logfile)
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            yield logpath
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def log_exception(logfile, exc_type, exc_value, exc_traceback):
    """Log an exception with its traceback to both stdout and the log file"""
    exc_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print('\n' + '=' * 80)
    print('Exception occurred:')
    print(exc_text)
    print('=' * 80)
    
    with open(logfile, 'a') as f:
        f.write('\n' + '=' * 80 + '\n')
        f.write('Exception occurred:\n')
        f.write(exc_text)
        f.write('=' * 80 + '\n')

# Run with output capture
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = Path("logs") / f"test_run_{timestamp}.log"

with capture_output(logfile.name) as logpath:
    print(f"Starting test run, logging to: {logfile}")
    
    try:
        import time
        
        # Set random seeds
        seed = int(time.time())  # Use current time as seed
        seed = 98324701398701328        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        attribute_metadata = [
            {
                'name': 'Gender',
                'type': 'categorical',
                'possible_values': [0, 1]  
            },
            {
                'name': 'Race',
                'type': 'categorical',
                'possible_values': [0, 1, 2, 3]  
            },
            {
                'name': 'Age',
                'type': 'categorical',
                'possible_values': [0, 1, 2, 3]  
            }
        ]

        # Load dataset and create graph
        dataset = AIFaceDataset("/home/brg2890/major/datasets/ai-face", ImageFileData, {}, AttributeNode, {"threshold": 2})
        dataloader = UnclusteredDeepfakeDataloader([dataset], Edge)
        train_graph, val_graph, test_graph = dataloader.load()
        
        # Create graph managers for each split
        train_manager = PerformanceGraphManager(
            graph=train_graph,
            rewire_threshold=0.8,
            edge_removal_threshold=0.2,
            max_edges_per_node=10,
            update_interval=200
        )
        val_manager = NoGraphManager(copy.deepcopy(val_graph))
        test_manager = NoGraphManager(copy.deepcopy(test_graph))

        # Define architectures to test
        cnn_architectures = [
            "swintransformdf",
            "resnestdf", 
            #"effnetdf",
            #"mesonetdf",
            #"squeezenetdf",
            #"vistransformdf",
            
        ]

        random.seed(13247987501)
        
        # Define traversal types to compare
        #traversal_types = ["comprehensive", "random", "i-value"]
        traversal_types = ["i-value"]
        
        # Test each architecture with both traversal types
        for arch in cnn_architectures:
            print(f"\n{'='*80}")
            print(f"Testing {arch} architecture")
            print(f"{'='*80}\n")
            
            for traversal_type in traversal_types:
                print(f"\n{'-'*40}")
                print(f"Using {traversal_type} traversal")
                print(f"{'-'*40}\n")
                
                try:
                    # Create model with adjusted learning rate
                    model = CNNModel(
                        f"/home/brg2890/major/bryce_python_workspace/GraphWork/HyperGraph/saved_models/{arch}_{traversal_type}_{timestamp}.pt",
                        arch,
                        0.001, 
                        True
                    )
                    
                    # Calculate appropriate number of steps
                    train_size = len(train_manager.graph.get_nodes())
                    val_size = len(val_manager.graph.get_nodes())
                    test_size = len(test_manager.graph.get_nodes())
                    
                    print(f"\nGraph sizes:")
                    print(f"Train: {train_size} nodes")
                    print(f"Val: {val_size} nodes")
                    print(f"Test: {test_size} nodes")
                    
                    # Set steps for each traversal type
                    train_steps = 2000  # Fixed number for training
                    val_steps = val_size  # Full validation set
                    test_steps = test_size  # Full test set
                    
                    # Create traversals - always use ComprehensiveTraversal for val and test
                    if traversal_type == 'i-value':
                        train_traversal = IValueTraversal(train_manager.graph, num_pointers=1, num_steps=train_steps)
                    elif traversal_type == 'random':
                        train_traversal = RandomTraversal(train_manager.graph, num_pointers=1, num_steps=train_steps)
                    else:  # comprehensive
                        train_traversal = ComprehensiveTraversal(train_manager.graph, num_pointers=1, num_steps=train_steps)

                    # Always use ComprehensiveTraversal for validation and test
                    val_traversal = ComprehensiveTraversal(val_manager.graph, num_pointers=1, num_steps=val_steps)
                    test_traversal = ComprehensiveTraversal(test_manager.graph, num_pointers=1, num_steps=test_steps)

                    print(f"\nTraversal settings:")
                    print(f"Train: {train_steps} steps with 1 pointer")
                    print(f"Val: {val_steps} steps with 1 pointer")
                    print(f"Test: {test_steps} steps with 1 pointer")
                    
                    # Create trainer
                    trainer = IValueTrainer(
                        graphmanager=train_manager,
                        train_traversal=train_traversal,
                        val_traversal=val_traversal,
                        models=[model],
                        attribute_metadata=attribute_metadata
                    )
                    
                    try:
                        print(f"Training {arch} with {traversal_type} traversal...")
                        trainer.run(num_epochs=5)
                        print(f"Testing {arch} with {traversal_type} traversal...")
                        test_metrics = trainer.test()
                        
                        # Handle both single dict and list of dicts for backwards compatibility
                        if isinstance(test_metrics, list):
                            for i, metrics in enumerate(test_metrics):
                                print(f"\nModel {i+1} Results:")
                                print(f"Loss: {metrics.get('avg_loss', 0.0):.4f}")
                                print(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
                                print(f"Bias Loss: {metrics.get('avg_bias_loss', 0.0):.4f}")
                        else:
                            print("\nTest Results:")
                            print(f"Loss: {test_metrics.get('avg_loss', 0.0):.4f}")
                            print(f"Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
                            print(f"Bias Loss: {test_metrics.get('avg_bias_loss', 0.0):.4f}")
                            
                        print(f"\nCompleted evaluation of {arch} with {traversal_type} traversal")
                    except Exception as e:
                        print(f"\nError while evaluating {arch} with {traversal_type}: {str(e)}")
                        log_exception(logfile, *sys.exc_info())
                        continue  # Continue with next configuration
                        
                except Exception as e:
                    print(f"\nError while setting up {arch} with {traversal_type}: {str(e)}")
                    log_exception(logfile, *sys.exc_info())
                    continue  # Continue with next configuration
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        log_exception(logfile, *sys.exc_info())
        sys.exit(1)
    except Exception:
        print("\nAn error occurred during execution")
        log_exception(logfile, *sys.exc_info())
        raise