from datetime import datetime
from pathlib import Path
import sys
from contextlib import contextmanager

from dataloaders.ClusteredDeepfakeDataloader import ClusteredDeepfakeDataloader
from datasets.CDFDataset import CDFDataset
from datasets.FFDataset import FFDataset
from data.ImageFileData import ImageFileData
from nodes.atrnode import AttributeNode
from nodes.RandomNode import RandomNode
from nodes.Node import Node
from edges.Edge import Edge
from managers.NoGraphManager import NoGraphManager

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

def main():
    # Run with output capture
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = Path("logs") / f"clustered_test_{timestamp}.log"

    with capture_output(logfile.name) as logpath:
        print(f"Starting test run, logging to: {logfile}")
        
        try:
            # Initialize datasets
            cdf_dataset = CDFDataset(
                "/home/brg2890/major/preprocessed/CelebDF-v2_new",
                ImageFileData,
                {},  # Empty dict for no transformations
                Node,
                {}
            )
            
            ff_dataset = FFDataset(
                "/home/brg2890/major/preprocessed/FaceForensics++_All/FaceForensics++",
                ImageFileData,
                {},  # Empty dict for no transformations
                Node,
                {}
            )
            
            # Initialize dataloader with both datasets and Edge class
            dataloader = ClusteredDeepfakeDataloader([cdf_dataset, ff_dataset], Edge)
            
            # Load the graph
            print("Loading and processing datasets...")
            graph = dataloader.load()
            
            # Create a manager (as done in test.py)
            manager = NoGraphManager(graph)
            
            print("Processing complete!")
            print(f"Total nodes: {len(graph.nodes)}")
            print(f"Total edges: {len(graph.edges)}")
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()