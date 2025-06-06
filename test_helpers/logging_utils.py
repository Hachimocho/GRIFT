import logging
import sys
import traceback
from pathlib import Path
from contextlib import contextmanager
import random
import numpy as np
import torch

class NullHandler(logging.Handler):
    def emit(self, record):
        pass

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
    
    # Ensure log_dir is created relative to the script or a defined base path
    # For simplicity, assuming 'logs' is a subdirectory in the current working directory
    # or where the main script is executed.
    log_dir = Path("logs") 
    log_dir.mkdir(exist_ok=True)
    
    logpath = log_dir / filename if not filename.startswith("logs/") else filename
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        with open(logpath, 'w') as logfile_handle:
            tee_stdout = TeeStream(old_stdout, logfile_handle)
            tee_stderr = TeeStream(old_stderr, logfile_handle)
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            yield logpath
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def log_exception(logfile_path, exc_type, exc_value, exc_traceback):
    """Log an exception with its traceback to both stdout and the log file"""
    exc_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print('\n' + '=' * 80)
    print('Exception occurred:')
    print(exc_text)
    print('=' * 80)
    
    try:
        with open(str(logfile_path), 'a') as f: # Ensure logfile_path is a string for open()
            f.write('\n' + '=' * 80 + '\n')
            f.write('Exception occurred:\n')
            f.write(exc_text)
            f.write('=' * 80 + '\n')
    except Exception as e:
        # Fallback if logging to file fails
        print(f"Critical: Failed to write exception to log file {logfile_path}: {e}", file=sys.stderr)

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Check if use_deterministic_algorithms is available (PyTorch 1.8+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        # For older PyTorch versions, the above might be enough, or specific operations might still be non-deterministic.
    print(f"Random seed set to {seed}")
