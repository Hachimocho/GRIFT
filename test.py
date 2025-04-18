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
            }
            # {
            #     'name': 'Age',
            #     'type': 'categorical',
            #     'possible_values': [0, 1, 2, 3]  
            # }
        ]

        # Load dataset and create graph
        # Done: Create nodes using dataset class
        dataset = AIFaceDataset("/home/brg2890/major/datasets/ai-face", ImageFileData, {}, AttributeNode, {"threshold": 2})
        # TODO: Finalize graph construction
        dataloader = UnclusteredDeepfakeDataloader([dataset], Edge)
        train_graph, val_graph, test_graph = dataloader.load()
        
        # Create graph managers for each split
        train_manager = NoGraphManager(copy.deepcopy(train_graph))
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
                    
                    # Create trainer based on traversal type
                    if traversal_type == "i-value":
                        trainer = IValueTrainer(
                            train_manager,  # Use train manager
                            None,  # Will be set by get_traversal
                            None,  # Will be set by get_traversal
                            [model],
                            attribute_metadata=attribute_metadata
                        )
                    else:  # random or comprehensive traversal
                        trainer = ExperimentTrainer(
                            train_manager,  # Use train manager
                            None,  # Will be set by get_traversal
                            None,  # Will be set by get_traversal
                            None,  # Will be set by get_traversal
                            [model],
                            traversal_type=traversal_type,
                            attribute_metadata=attribute_metadata
                        )
                    
                    # Create traversals for each split
                    # Use more pointers and adjust steps based on graph sizes
                    train_size = len(train_manager.graph.get_nodes())
                    val_size = len(val_manager.graph.get_nodes())
                    test_size = len(test_manager.graph.get_nodes())
                    
                    print(f"\nGraph sizes:")
                    print(f"Train: {train_size} nodes")
                    print(f"Val: {val_size} nodes")
                    print(f"Test: {test_size} nodes")
                    
                    # Calculate appropriate number of steps
                    train_steps = 2000
                    val_steps = 1000 
                    test_steps = None  # Use None to visit all test nodes
                    
                    if traversal_type == "comprehensive":
                        train_traversal = ComprehensiveTraversal(train_manager.graph, num_pointers=1, num_steps=train_steps)
                    else:
                        train_traversal = trainer.get_traversal(train_manager.graph, num_pointers=1, num_steps=train_steps)
                    val_traversal = ComprehensiveTraversal(val_manager.graph, num_pointers=1, num_steps=val_steps)
                    test_traversal = ComprehensiveTraversal(test_manager.graph, num_pointers=1, num_steps=test_steps)
                    
                    print(f"\nTraversal settings:")
                    print(f"Train: {train_steps} steps with 1 pointers")
                    print(f"Val: {val_steps} steps with 1 pointers")
                    print(f"Test: All nodes with 1 pointers")
                    
                    # Update trainer with correct traversals
                    trainer.train_traversal = train_traversal
                    trainer.val_traversal = val_traversal
                    trainer.test_traversal = test_traversal
                    
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