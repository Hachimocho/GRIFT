import wandb
import sys
import traceback
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from trainers.IValueTrainer import IValueTrainer
from dataloaders.UnclusteredDeepfakeDataloader import UnclusteredDeepfakeDataloader
from datasets.AIFaceDataset import AIFaceDataset
from data.ImageFileData import ImageFileData
from nodes.atrnode import AttributeNode
from managers.NoGraphManager import NoGraphManager
from traversals.ComprehensiveTraversal import ComprehensiveTraversal
from traversals.RandomTraversal import RandomTraversal
from models.CNNModel import CNNModel
from edges.Edge import Edge

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
        num_models = 1
        
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

        dataloader = UnclusteredDeepfakeDataloader(
            [AIFaceDataset("/home/brg2890/major/datasets/ai-face", ImageFileData, {}, AttributeNode, {"threshold": 2})],
            Edge
        )
        graph = dataloader.load()
        manager = NoGraphManager(graph)
        train_traversal = RandomTraversal(manager.graph, num_models, 300)
        test_traversal = ComprehensiveTraversal(manager.graph, num_models)
        models = [CNNModel("/home/brg2890/major/bryce_python_workspace/GraphWork/HyperGraph/saved_models/test_" + str(i) + ".pt", "resnestdf", 0.001, True) for i in range(num_models)]
        
        trainer = IValueTrainer(manager, train_traversal, test_traversal, models, attribute_metadata)
        trainer.run()
        trainer.test_run()
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        log_exception(logfile, *sys.exc_info())
        sys.exit(1)
    except Exception:
        print("\nAn error occurred during execution")
        log_exception(logfile, *sys.exc_info())
        raise  