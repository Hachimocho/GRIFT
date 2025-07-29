#!/usr/bin/env python3
"""
GPU Queue Manager for HyperGraph Test UI

Manages GPU allocation and queueing for test runs. Provides functionality to:
- Check GPU availability
- Allocate GPUs to runs
- Queue runs when no GPUs are available
- Monitor GPU usage
- Handle run scheduling and execution

Author: Quanty 7
"""

import os
import json
import time
import threading
import subprocess
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid
import psutil
import logging

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Warning: GPUtil not available. GPU management will be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GPU detection may be limited.")

logger = logging.getLogger(__name__)

class GPUQueueManager:
    """Manages GPU allocation and queueing for test runs."""
    
    def __init__(self, runs_dir: str = "web_ui/runs"):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU management
        self.gpu_allocations = {}  # {gpu_id: run_id}
        self.gpu_processes = {}    # {gpu_id: process}
        self.gpu_monitor_threads = {}  # {gpu_id: thread}
        
        # Queue management
        self.run_queue = []  # List of (run_id, config_name, config, priority)
        self.queue_lock = threading.Lock()
        
        # Run tracking
        self.active_runs = {}  # {run_id: metadata}
        self.run_metadata = {}  # {run_id: full_metadata}
        
        # Configuration
        self.min_gpu_memory_gb = 2.0  # Minimum GPU memory required
        self.gpu_check_interval = 5.0  # Seconds between GPU availability checks
        self.queue_check_interval = 2.0  # Seconds between queue processing
        
        # Threading
        self.running = True
        self.queue_thread = None
        self.gpu_monitor_thread = None
        
        # Start background threads
        self._start_background_threads()
        
        # Reconcile existing runs on startup
        self.reconcile_existing_runs()
    
    def _start_background_threads(self):
        """Start background threads for queue processing and GPU monitoring."""
        # Queue processing thread
        self.queue_thread = threading.Thread(target=self._process_queue_loop, daemon=True)
        self.queue_thread.start()
        
        # GPU monitoring thread
        self.gpu_monitor_thread = threading.Thread(target=self._monitor_gpus_loop, daemon=True)
        self.gpu_monitor_thread.start()
        
        logger.info("GPU Queue Manager background threads started")
    
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get information about all available GPUs."""
        gpu_info = []
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_data = {
                        'id': i,
                        'name': gpu.name,
                        'memory_total_gb': gpu.memoryTotal / 1024,
                        'memory_used_gb': gpu.memoryUsed / 1024,
                        'memory_free_gb': gpu.memoryFree / 1024,
                        'memory_utilization': gpu.memoryUtil * 100,
                        'temperature': gpu.temperature,
                        'load': gpu.load * 100 if gpu.load else 0,
                        'allocated_to': self.gpu_allocations.get(i),
                        'status': 'allocated' if i in self.gpu_allocations else 'available'
                    }
                    gpu_info.append(gpu_data)
            except Exception as e:
                logger.error(f"Error getting GPU info with GPUtil: {e}")
        
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            # Fallback to PyTorch GPU detection
            try:
                for i in range(torch.cuda.device_count()):
                    gpu_data = {
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                        'memory_used_gb': torch.cuda.memory_allocated(i) / (1024**3),
                        'memory_free_gb': (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / (1024**3),
                        'memory_utilization': (torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory) * 100,
                        'temperature': None,
                        'load': None,
                        'allocated_to': self.gpu_allocations.get(i),
                        'status': 'allocated' if i in self.gpu_allocations else 'available'
                    }
                    gpu_info.append(gpu_data)
            except Exception as e:
                logger.error(f"Error getting GPU info with PyTorch: {e}")
        
        return gpu_info
    
    def get_available_gpus(self, min_memory_gb: float = None) -> List[int]:
        """Get list of available GPU IDs with sufficient memory."""
        if min_memory_gb is None:
            min_memory_gb = self.min_gpu_memory_gb
        
        available_gpus = []
        gpu_info = self.get_gpu_info()
        
        for gpu in gpu_info:
            if (gpu['status'] == 'available' and 
                gpu['memory_free_gb'] >= min_memory_gb):
                available_gpus.append(gpu['id'])
        
        return available_gpus
    
    def estimate_gpu_memory_requirement(self, config: Dict[str, Any]) -> float:
        """Estimate GPU memory requirement for a configuration."""
        # Base memory requirement
        base_memory = 2.0  # GB
        
        # Adjust based on batch size
        batch_size = config.get('batch_size', 32)
        if batch_size > 64:
            base_memory += 2.0
        elif batch_size > 32:
            base_memory += 1.0
        
        # Adjust based on model architecture
        architectures = config.get('architectures', ['resnestdf'])
        for arch in architectures:
            if 'swin' in arch.lower() or 'transformer' in arch.lower():
                base_memory += 2.0
            elif 'resnest' in arch.lower() or 'resnet' in arch.lower():
                base_memory += 1.0
        
        # Adjust based on number of epochs
        num_epochs = config.get('num_epochs', 10)
        if num_epochs > 50:
            base_memory += 1.0
        
        return min(base_memory, 8.0)  # Cap at 8GB
    
    def queue_run(self, config_name: str, config: Dict[str, Any], priority: int = 0) -> str:
        """Add a run to the queue."""
        run_id = self._generate_run_id()
        
        # Create run metadata
        metadata = {
            "run_id": run_id,
            "config_name": config_name,
            "config": config,
            "priority": priority,
            "queued_time": datetime.now().isoformat(),
            "status": "queued",
            "estimated_gpu_memory": self.estimate_gpu_memory_requirement(config),
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Save metadata
        self._save_run_metadata(run_id, metadata)
        self.run_metadata[run_id] = metadata
        
        # Add to queue
        with self.queue_lock:
            self.run_queue.append((run_id, config_name, config, priority))
            # Sort by priority (higher priority first)
            self.run_queue.sort(key=lambda x: x[3], reverse=True)
        
        logger.info(f"Queued run {run_id} with priority {priority}")
        return run_id
    
    def start_run(self, run_id: str, gpu_id: int) -> bool:
        """Start a specific run on a specific GPU."""
        try:
            metadata = self.run_metadata.get(run_id)
            if not metadata:
                logger.error(f"Run {run_id} not found in metadata")
                return False
            
            config = metadata['config']
            
            # Build command arguments
            cmd_args = self._build_command_args(config, run_id, gpu_id)
            
            # Create log file
            log_file = self.runs_dir / f"{run_id}.log"
            
            # Write debug information to log file
            with open(log_file, 'w') as f:
                f.write(f"=== Test Run Debug Information ===\n")
                f.write(f"Run ID: {run_id}\n")
                f.write(f"GPU ID: {gpu_id}\n")
                f.write(f"Config Name: {metadata['config_name']}\n")
                f.write(f"Configuration: {json.dumps(config, indent=2)}\n")
                f.write(f"Command: {' '.join(cmd_args)}\n")
                f.write(f"=== End Debug Information ===\n\n")
            
            # Update metadata
            metadata.update({
                "status": "running",
                "gpu_id": gpu_id,
                "start_time": datetime.now().isoformat(),
                "log_file": str(log_file),
                "command": " ".join(cmd_args),
                "last_updated": datetime.now().isoformat()
            })
            
            # Save updated metadata
            self._save_run_metadata(run_id, metadata)
            
            # Start process with GPU restriction
            with open(log_file, 'a') as f:
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                
                process = subprocess.Popen(
                    cmd_args,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    env=env,
                    preexec_fn=os.setsid
                )
            
            # Track process and allocation
            self.gpu_allocations[gpu_id] = run_id
            self.gpu_processes[gpu_id] = process
            self.active_runs[run_id] = metadata
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_run_process,
                args=(run_id, process, gpu_id, str(log_file))
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            self.gpu_monitor_threads[gpu_id] = monitor_thread
            
            logger.info(f"Started run {run_id} on GPU {gpu_id} with PID {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting run {run_id}: {e}")
            return False
    
    def stop_run(self, run_id: str) -> bool:
        """Stop a running test."""
        try:
            # Find which GPU this run is using
            gpu_id = None
            for gid, rid in self.gpu_allocations.items():
                if rid == run_id:
                    gpu_id = gid
                    break
            
            if gpu_id is None:
                logger.warning(f"Run {run_id} not found in active runs")
                return False
            
            # Stop the process
            process = self.gpu_processes.get(gpu_id)
            if process:
                try:
                    # Send SIGTERM to the process group
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    
                    # Wait a bit, then force kill if needed
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait()
                    
                    logger.info(f"Stopped run {run_id} on GPU {gpu_id}")
                except Exception as e:
                    logger.error(f"Error stopping process for run {run_id}: {e}")
            
            # Clean up GPU allocation
            self._release_gpu(gpu_id, run_id, "stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping run {run_id}: {e}")
            return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self.queue_lock:
            queue_info = []
            for run_id, config_name, config, priority in self.run_queue:
                metadata = self.run_metadata.get(run_id, {})
                queue_info.append({
                    'run_id': run_id,
                    'config_name': config_name,
                    'priority': priority,
                    'queued_time': metadata.get('queued_time'),
                    'estimated_gpu_memory': metadata.get('estimated_gpu_memory', 0)
                })
        
        return {
            'queue_length': len(self.run_queue),
            'queued_runs': queue_info,
            'active_runs': list(self.active_runs.keys()),
            'available_gpus': self.get_available_gpus(),
            'gpu_allocations': self.gpu_allocations.copy()
        }
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all runs with their current status."""
        runs = []
        
        # Add queued runs
        with self.queue_lock:
            for run_id, config_name, config, priority in self.run_queue:
                metadata = self.run_metadata.get(run_id, {})
                runs.append({
                    'run_id': run_id,
                    'config_name': config_name,
                    'status': 'queued',
                    'priority': priority,
                    'queued_time': metadata.get('queued_time'),
                    'created': metadata.get('created', metadata.get('queued_time', '')),
                    'last_updated': metadata.get('last_updated', metadata.get('queued_time', ''))
                })
        
        # Add active runs
        for run_id, metadata in self.active_runs.items():
            runs.append({
                'run_id': run_id,
                'config_name': metadata.get('config_name'),
                'status': 'running',
                'gpu_id': metadata.get('gpu_id'),
                'start_time': metadata.get('start_time'),
                'created': metadata.get('created', metadata.get('start_time', '')),
                'last_updated': metadata.get('last_updated', metadata.get('start_time', ''))
            })
        
        # Add completed/failed runs from files
        for run_file in self.runs_dir.glob("*.json"):
            if run_file.stem.startswith("run_"):
                run_id = run_file.stem
                if run_id not in [r['run_id'] for r in runs]:
                    metadata = self._load_run_metadata(run_id)
                    if metadata:
                        run_info = {
                            'run_id': run_id,
                            'config_name': metadata.get('config_name'),
                            'status': metadata.get('status', 'unknown'),
                            'start_time': metadata.get('start_time'),
                            'end_time': metadata.get('end_time'),
                            'created': metadata.get('created', metadata.get('start_time', '')),
                            'last_updated': metadata.get('last_updated', metadata.get('end_time', ''))
                        }
                        
                        # Include results for completed runs
                        if metadata.get('status') == 'completed' and 'results' in metadata:
                            run_info['results'] = metadata['results']
                            logger.info(f"DEBUG: Added results for {run_id}: {metadata['results']}")
                        elif metadata.get('status') == 'completed':
                            logger.info(f"DEBUG: Completed run {run_id} has no results field")
                        
                        runs.append(run_info)
        
        return runs
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific run."""
        # Check active runs first
        if run_id in self.active_runs:
            return self.active_runs[run_id]
        
        # Check metadata
        if run_id in self.run_metadata:
            return self.run_metadata[run_id]
        
        # Check saved metadata
        return self._load_run_metadata(run_id)
    
    def get_run_logs(self, run_id: str, tail_lines: int = 0) -> List[str]:
        """Get logs for a specific run."""
        metadata = self.get_run(run_id)
        if not metadata:
            return []
        
        log_file = metadata.get('log_file')
        if not log_file or not os.path.exists(log_file):
            return []
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            if tail_lines > 0:
                lines = lines[-tail_lines:]
            
            return lines
        except Exception as e:
            logger.error(f"Error reading logs for run {run_id}: {e}")
            return []
    
    def _process_queue_loop(self):
        """Background thread to process the run queue."""
        while self.running:
            try:
                with self.queue_lock:
                    if self.run_queue:
                        run_id, config_name, config, priority = self.run_queue[0]
                        
                        # Check if we have an available GPU
                        required_memory = self.estimate_gpu_memory_requirement(config)
                        available_gpus = self.get_available_gpus(required_memory)
                        
                        if available_gpus:
                            # Start the run on the first available GPU
                            gpu_id = available_gpus[0]
                            if self.start_run(run_id, gpu_id):
                                # Remove from queue
                                self.run_queue.pop(0)
                                logger.info(f"Started queued run {run_id} on GPU {gpu_id}")
                            else:
                                logger.error(f"Failed to start queued run {run_id}")
                
                time.sleep(self.queue_check_interval)
                
            except Exception as e:
                logger.error(f"Error in queue processing loop: {e}")
                time.sleep(self.queue_check_interval)
    
    def _monitor_gpus_loop(self):
        """Background thread to monitor GPU usage and detect completed runs."""
        orphaned_check_counter = 0
        orphaned_check_interval = 30  # Check for orphaned runs every 30 seconds
        
        while self.running:
            try:
                # Check for completed processes
                completed_gpus = []
                for gpu_id, process in self.gpu_processes.items():
                    if process.poll() is not None:
                        completed_gpus.append(gpu_id)
                
                # Handle completed runs
                for gpu_id in completed_gpus:
                    run_id = self.gpu_allocations.get(gpu_id)
                    if run_id:
                        exit_code = self.gpu_processes[gpu_id].returncode
                        status = "completed" if exit_code == 0 else "failed"
                        self._release_gpu(gpu_id, run_id, status, exit_code=exit_code)
                
                # Periodically check for orphaned queued runs
                orphaned_check_counter += 1
                if orphaned_check_counter >= orphaned_check_interval:
                    self.check_orphaned_queued_runs()
                    orphaned_check_counter = 0
                
                time.sleep(self.gpu_check_interval)
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {e}")
                time.sleep(self.gpu_check_interval)
    
    def _monitor_run_process(self, run_id: str, process: subprocess.Popen, gpu_id: int, log_file: str):
        """Monitor a specific run process."""
        try:
            # Wait for process to complete
            exit_code = process.wait()
            
            # Determine status
            status = "completed" if exit_code == 0 else "failed"
            
            # Release GPU
            self._release_gpu(gpu_id, run_id, status, exit_code=exit_code)
            
        except Exception as e:
            logger.error(f"Error monitoring run {run_id}: {e}")
            self._release_gpu(gpu_id, run_id, "failed", error=str(e))
    
    def _release_gpu(self, gpu_id: int, run_id: str, status: str, exit_code: int = None, error: str = None):
        """Release a GPU and update run status."""
        try:
            # Update run metadata
            metadata = self.run_metadata.get(run_id)
            if metadata:
                metadata.update({
                    "status": status,
                    "end_time": datetime.now().isoformat(),
                    "exit_code": exit_code,
                    "last_updated": datetime.now().isoformat()
                })
                if error:
                    metadata["error"] = error
                
                # Extract results if run completed successfully
                if status == "completed":
                    self._extract_results(run_id)
                
                self._save_run_metadata(run_id, metadata)
            
            # Clean up GPU allocation
            if gpu_id in self.gpu_allocations:
                del self.gpu_allocations[gpu_id]
            if gpu_id in self.gpu_processes:
                del self.gpu_processes[gpu_id]
            if gpu_id in self.gpu_monitor_threads:
                del self.gpu_monitor_threads[gpu_id]
            if run_id in self.active_runs:
                del self.active_runs[run_id]
            
            logger.info(f"Released GPU {gpu_id} from run {run_id} (status: {status})")
            
        except Exception as e:
            logger.error(f"Error releasing GPU {gpu_id}: {e}")
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{short_uuid}"
    
    def _build_command_args(self, config: Dict[str, Any], run_id: str = None, gpu_id: int = None) -> List[str]:
        """Build command line arguments from configuration."""
        args = ["python", "test_hierarchical.py"]
        
        # Define argument mapping
        arg_mapping = {
            "test": "--test",
            "visualize": "--visualize",
            "show": "--show",
            "search": "--search",
            "search_split": "--search-split",
            "quality_steps": "--quality-steps",
            "symmetry_steps": "--symmetry-steps",
            "embedding_steps": "--embedding-steps",
            "search_results": "--search-results",
            "traversal_type": "--traversal-type",
            "enable_traversal_switching": "--enable-traversal-switching",
            "traversal_sequence": "--traversal-sequence",
            "switch_epochs": "--switch-epochs",
            "test_all_traversals": "--test-all-traversals",
            "architectures": "--architectures",
            "num_epochs": "--num-epochs",
            "batch_size": "--batch-size",
            "bias_hop_period": "--bias_hop_period",
            "seed": "--seed",
            "quality_threshold": "--quality-threshold",
            "symmetry_threshold": "--symmetry-threshold",
            "embedding_threshold": "--embedding-threshold",
            "cached_nodes": "--use-cached",
            "cache_nodes": "--cache-nodes",
            "cached_nodes_count": "--cached-nodes",
            "fair_train": "--fair-train",
            "fair_test": "--fair-test",
            "enable_ivalue_viz": "--enable-ivalue-viz",
            "viz_track_nodes": "--viz-track-nodes",
            "viz_sample_size": "--viz-sample-size",
            "viz_save_dir": "--viz-save-dir",
            "bias_loss_weight": "--bias_loss_weight",
            "num_workers": "--num-workers",
            "dqn_model": "--dqn-model",
            "graph_type": "--graph-type"
        }
        
        # Add arguments based on configuration
        for config_key, arg_name in arg_mapping.items():
            if config_key in config:
                value = config[config_key]
                
                # Handle boolean flags
                if isinstance(value, bool):
                    if value:
                        args.append(arg_name)
                else:
                    args.extend([arg_name, str(value)])
        
        # Special handling for cache flags
        if config.get("cached_nodes", False):
            args.append("--use-cached")
        
        # Add run ID if provided
        if run_id:
            args.extend(["--run-id", run_id])
        
        return args
    
    def _save_run_metadata(self, run_id: str, metadata: Dict[str, Any]) -> bool:
        """Save run metadata to file."""
        try:
            metadata_file = self.runs_dir / f"{run_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving run metadata for {run_id}: {e}")
            return False
    
    def _load_run_metadata(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load run metadata from file."""
        try:
            metadata_file = self.runs_dir / f"{run_id}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading run metadata for {run_id}: {e}")
        return None
    
    def reconcile_existing_runs(self):
        """Reconcile existing runs on startup."""
        try:
            # Check for any runs marked as running or queued
            for run_file in self.runs_dir.glob("*.json"):
                if run_file.stem.startswith("run_"):
                    metadata = self._load_run_metadata(run_file.stem)
                    if metadata:
                        status = metadata.get("status")
                        if status == "running":
                            # Mark as failed since we don't know the actual status
                            metadata.update({
                                "status": "failed",
                                "end_time": datetime.now().isoformat(),
                                "error": "Process status unknown on server restart"
                            })
                            self._save_run_metadata(run_file.stem, metadata)
                            logger.info(f"Marked orphaned running run {run_file.stem} as failed")
                        elif status == "queued":
                            # Mark as failed since we don't know if it's still in queue
                            metadata.update({
                                "status": "failed",
                                "end_time": datetime.now().isoformat(),
                                "error": "Queue status unknown on server restart"
                            })
                            self._save_run_metadata(run_file.stem, metadata)
                            logger.info(f"Marked orphaned queued run {run_file.stem} as failed")
        except Exception as e:
            logger.error(f"Error reconciling existing runs: {e}")
    
    def shutdown(self):
        """Shutdown the GPU queue manager."""
        self.running = False
        
        # Stop all running processes
        for gpu_id, process in self.gpu_processes.items():
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except Exception as e:
                logger.error(f"Error stopping process on GPU {gpu_id}: {e}")
        
        logger.info("GPU Queue Manager shutdown complete")
    
    def _extract_results(self, run_id: str):
        """Extract results from completed test run."""
        logger.info(f"DEBUG: _extract_results called for run {run_id}")
        try:
            metadata = self.run_metadata.get(run_id)
            if not metadata:
                # Try to load from file
                metadata = self._load_run_metadata(run_id)
                if not metadata:
                    logger.warning(f"Could not load metadata for run {run_id}")
                    return
            
            # Try to extract final accuracy and bias metrics from logs
            log_file = metadata.get('log_file')
            logger.info(f"DEBUG: log_file for {run_id}: {log_file}")
            logger.info(f"DEBUG: log_file exists: {log_file and os.path.exists(log_file)}")
            if log_file and os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    logger.info(f"DEBUG: log_content length for {run_id}: {len(log_content)}")
                    
                    # Initialize results dict if not present
                    if "results" not in metadata:
                        metadata["results"] = {}
                    
                    # Look for final test results
                    if "Final Test Results" in log_content:
                        # Extract accuracy from log using regex
                        import re
                        acc_match = re.search(r'Final Test Results: Accuracy=([0-9.]+)%', log_content)
                        if acc_match:
                            # Store as decimal (0.8044) instead of percentage (80.44)
                            metadata["results"]["final_accuracy"] = float(acc_match.group(1)) / 100.0
                            logger.info(f"Extracted accuracy {acc_match.group(1)}% (stored as {float(acc_match.group(1)) / 100.0}) for run {run_id}")
                        else:
                            logger.warning(f"Could not extract accuracy from log for run {run_id}")
                        
                        # Extract loss if available (from Final Test Results)
                        loss_match = re.search(r'Final Test Results: Accuracy=[0-9.]+%, Avg Loss=([0-9.]+)', log_content)
                        if loss_match:
                            metadata["results"]["loss"] = float(loss_match.group(1))
                            logger.info(f"Extracted loss {loss_match.group(1)} for run {run_id}")
                        
                        # Extract duration if available (look for various time patterns)
                        duration_match = re.search(r'Training completed in\s*([0-9.]+)\s*seconds', log_content)
                        if not duration_match:
                            duration_match = re.search(r'Total time:\s*([0-9.]+)\s*seconds', log_content)
                        if not duration_match:
                            duration_match = re.search(r'Elapsed time:\s*([0-9.]+)\s*seconds', log_content)
                        if duration_match:
                            duration_seconds = float(duration_match.group(1))
                            # Convert to human readable format
                            if duration_seconds < 60:
                                duration_str = f"{duration_seconds:.1f}s"
                            elif duration_seconds < 3600:
                                duration_str = f"{duration_seconds/60:.1f}m"
                            else:
                                duration_str = f"{duration_seconds/3600:.1f}h"
                            metadata["results"]["duration"] = duration_str
                            logger.info(f"Extracted duration {duration_str} for run {run_id}")
                        
                        # Extract architecture and traversal from configuration section
                        config_match = re.search(r'"architectures":\s*"([^"]+)"', log_content)
                        if config_match:
                            metadata["results"]["architecture"] = config_match.group(1)
                            logger.info(f"Extracted architecture {config_match.group(1)} for run {run_id}")
                        
                        traversal_match = re.search(r'"traversal_type":\s*"([^"]+)"', log_content)
                        if traversal_match:
                            metadata["results"]["traversal_type"] = traversal_match.group(1)
                            logger.info(f"Extracted traversal_type {traversal_match.group(1)} for run {run_id}")
                    else:
                        logger.warning(f"No 'Final Test Results' found in log for run {run_id}")
                    
                    # Always try to extract bias metrics, regardless of whether accuracy was found
                    # Extract bias metrics using simple regex patterns
                    # Look for the specific bias values in the log
                    
                    # Debug: check if bias_metrics section exists
                    if '"bias_metrics"' in log_content:
                        logger.info(f"Found bias_metrics section in log for {run_id}")
                        # Find the bias_metrics section
                        bias_start = log_content.find('"bias_metrics"')
                        bias_section = log_content[bias_start:min(bias_start + 2000, len(log_content))]
                        logger.info(f"Bias section preview: {bias_section[:500]}")
                    else:
                        logger.warning(f"No bias_metrics section found in log for {run_id}")
                    
                    race_gender_match = re.search(r'"race_gender_overall_bias":\s*([0-9.]+)', log_content)
                    gender_match = re.search(r'"Ground Truth Gender":\s*([0-9.]+)', log_content)
                    race_match = re.search(r'"Ground Truth Race":\s*([0-9.]+)', log_content)
                    avg_bias_match = re.search(r'"average_attribute_bias":\s*([0-9.]+)', log_content)
                    
                    # Debug: log what we found
                    logger.info(f"Bias extraction debug for {run_id}:")
                    logger.info(f"  race_gender_match: {race_gender_match}")
                    logger.info(f"  gender_match: {gender_match}")
                    logger.info(f"  race_match: {race_match}")
                    logger.info(f"  avg_bias_match: {avg_bias_match}")
                    
                    # Also try looking for the values in the per_attribute_bias section
                    if not gender_match:
                        gender_match = re.search(r'"per_attribute_bias":\s*{[^}]*"Ground Truth Gender":\s*([0-9.]+)', log_content)
                    if not race_match:
                        race_match = re.search(r'"per_attribute_bias":\s*{[^}]*"Ground Truth Race":\s*([0-9.]+)', log_content)
                    
                    if race_gender_match or gender_match or race_match or avg_bias_match:
                        # Extract bias metrics
                        if race_gender_match:
                            metadata["results"]["race_gender_bias"] = float(race_gender_match.group(1))
                        if gender_match:
                            metadata["results"]["gender_bias"] = float(gender_match.group(1))
                        if race_match:
                            metadata["results"]["race_bias"] = float(race_match.group(1))
                        if avg_bias_match:
                            metadata["results"]["average_attribute_bias"] = float(avg_bias_match.group(1))
                        
                        logger.info(f"Extracted bias metrics for run {run_id}: race_gender={metadata['results'].get('race_gender_bias')}, gender={metadata['results'].get('gender_bias')}, race={metadata['results'].get('race_bias')}, avg={metadata['results'].get('average_attribute_bias')}")
                    else:
                        logger.warning(f"No bias metrics found in log for run {run_id}")
                        
                except Exception as e:
                    logger.error(f"Error extracting results from log for {run_id}: {e}")
            else:
                logger.warning(f"Log file not found for run {run_id}: {log_file}")
            
            # Save updated metadata
            self._save_run_metadata(run_id, metadata)
            
        except Exception as e:
            logger.error(f"Error extracting results for {run_id}: {e}")
    
    def _analyze_run_status_from_log(self, run_id: str) -> Optional[str]:
        """Analyze run status from log file to detect failed runs that exited with code 0."""
        try:
            import re
            
            metadata = self.run_metadata.get(run_id)
            if not metadata:
                metadata = self._load_run_metadata(run_id)
                if not metadata:
                    return None
            
            log_file = metadata.get('log_file')
            if not log_file:
                logger.warning(f"No log file path found for run {run_id}")
                return "failed"  # No log file means the run failed
            
            if not os.path.exists(log_file):
                logger.warning(f"Log file not found for run {run_id}: {log_file}")
                return "failed"  # Missing log file means the run failed
            
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Check for error patterns that indicate failure
            error_patterns = [
                r'Exception occurred:',
                r'Error in configuration',
                r'Traceback \(most recent call last\):',
                r'AttributeError:',
                r'RuntimeError:',
                r'ValueError:',
                r'ImportError:',
                r'ModuleNotFoundError:',
                r'FileNotFoundError:',
                r'PermissionError:',
                r'MemoryError:',
                r'CUDA out of memory',
                r'Killed',
                r'Segmentation fault'
            ]
            
            for pattern in error_patterns:
                if re.search(pattern, log_content, re.IGNORECASE):
                    logger.info(f"Detected error pattern '{pattern}' in log for run {run_id}")
                    return "failed"
            
            # Check if the run actually completed successfully
            if "Final Test Results" in log_content:
                return "completed"
            
            # If no clear success or failure indicators, return None (unknown)
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing run status from log for {run_id}: {e}")
            return None
    
    def check_orphaned_queued_runs(self) -> List[str]:
        """Check for runs marked as 'queued' that are no longer in the actual queue and mark them as failed."""
        orphaned_runs = []
        
        try:
            # Get the actual queue
            with self.queue_lock:
                actual_queue_run_ids = [run_id for run_id, _, _, _ in self.run_queue]
            
            # Check all run metadata for runs marked as queued but not in actual queue
            orphaned_run_ids = []
            for run_id, metadata in self.run_metadata.items():
                if metadata.get('status') == 'queued' and run_id not in actual_queue_run_ids:
                    orphaned_run_ids.append(run_id)
            
            # Also check saved files for runs marked as queued but not in actual queue
            for run_file in self.runs_dir.glob("*.json"):
                if run_file.stem.startswith("run_"):
                    run_id = run_file.stem
                    metadata = self._load_run_metadata(run_id)
                    if metadata and metadata.get('status') == 'queued' and run_id not in actual_queue_run_ids:
                        if run_id not in orphaned_run_ids:
                            orphaned_run_ids.append(run_id)
                        # Also update the in-memory metadata if it exists
                        if run_id in self.run_metadata:
                            self.run_metadata[run_id] = metadata
            
            # Debug logging
            logger.info(f"Checking for orphaned runs: {len(orphaned_run_ids)} runs marked as queued but not in actual queue")
            logger.info(f"Actual queue run IDs: {actual_queue_run_ids}")
            logger.info(f"Orphaned run IDs: {orphaned_run_ids}")
            
            # Mark orphaned runs as failed
            for run_id in orphaned_run_ids:
                metadata = self.run_metadata.get(run_id)
                if metadata:
                    logger.info(f"Marking orphaned run {run_id} as failed")
                    metadata.update({
                        "status": "failed",
                        "end_time": datetime.now().isoformat(),
                        "error": "Run was orphaned from queue - no longer in actual queue",
                        "last_updated": datetime.now().isoformat()
                    })
                    self._save_run_metadata(run_id, metadata)
                    orphaned_runs.append(run_id)
                    logger.warning(f"Marked orphaned queued run {run_id} as failed")
                else:
                    logger.warning(f"Could not find metadata for orphaned run {run_id}")
            
            if orphaned_runs:
                logger.info(f"Found and marked {len(orphaned_runs)} orphaned queued runs as failed")
            else:
                logger.info("No orphaned runs found")
            
        except Exception as e:
            logger.error(f"Error checking for orphaned queued runs: {e}")
        
        return orphaned_runs 