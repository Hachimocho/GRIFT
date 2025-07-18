#!/usr/bin/env python3
"""
Test Runner for HyperGraph Test UI

Manages test execution, monitoring, and results collection for the HyperGraph
deepfake detection system. Handles background test runs and log collection.

Author: Quanty 7
"""

import os
import json
import subprocess
import threading
import time
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
import psutil

class TestRunner:
    """Manages test execution and monitoring."""
    
    def __init__(self, runs_dir: str = "web_ui/runs"):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Track active processes
        self.active_processes = {}
        self.process_threads = {}
        
        # Reconcile run statuses on startup
        self.reconcile_run_statuses()
    
    def reconcile_run_statuses(self):
        """Check all runs with status 'running' and update if process is not alive."""
        for run_file in self.runs_dir.glob("*.json"):
            if run_file.stem.startswith("run_"):
                metadata = self._load_run_metadata(run_file.stem)
                if metadata and metadata.get("status") == "running":
                    pid = None
                    # Try to get PID from metadata (if present)
                    if "pid" in metadata:
                        pid = metadata["pid"]
                    else:
                        # Try to get from active_processes (should be empty on fresh start)
                        pass
                    process_alive = False
                    if pid is not None:
                        try:
                            import psutil
                            p = psutil.Process(pid)
                            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                                process_alive = True
                        except Exception:
                            process_alive = False
                    # If process is not alive, update status
                    if not process_alive:
                        self._update_run_status(
                            metadata["run_id"],
                            "failed",
                            end_time=datetime.now().isoformat(),
                            error="Process not found on server startup. Marked as failed."
                        )
    
    def start_run(self, config_name: str, config: Dict[str, Any]) -> Optional[str]:
        """Start a test run with the given configuration."""
        try:
            run_id = self._generate_run_id()
            
            # Build command arguments, passing run_id
            cmd_args = self._build_command_args(config, run_id)
            
            # Create log file
            log_file = self.runs_dir / f"{run_id}.log"
            
            # Write debug information to log file
            with open(log_file, 'w') as f:
                f.write(f"=== Test Run Debug Information ===\n")
                f.write(f"Run ID: {run_id}\n")
                f.write(f"Config Name: {config_name}\n")
                f.write(f"Configuration: {json.dumps(config, indent=2)}\n")
                f.write(f"Command: {' '.join(cmd_args)}\n")
                f.write(f"=== End Debug Information ===\n\n")
            
            # Create run metadata
            metadata = {
                "run_id": run_id,
                "config_name": config_name,
                "config": config,
                "command": " ".join(cmd_args),
                "start_time": datetime.now().isoformat(),
                "status": "running",
                "log_file": str(log_file),
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            
            # Save metadata
            if not self._save_run_metadata(run_id, metadata):
                return None
            
            # Start process
            with open(log_file, 'a') as f:
                process = subprocess.Popen(
                    cmd_args,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    preexec_fn=os.setsid  # Create new process group for easier termination
                )
            
            # Track process
            self.active_processes[run_id] = process
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(run_id, process, str(log_file))
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            self.process_threads[run_id] = monitor_thread
            
            print(f"Started test run {run_id} with PID {process.pid}")
            return run_id
            
        except Exception as e:
            print(f"Error starting test run: {e}")
            return None
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{short_uuid}"
    
    def _build_command_args(self, config: Dict[str, Any], run_id: str = None) -> List[str]:
        """Build command line arguments from configuration. Always appends --run-id <run_id>."""
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
        
        # Add arguments based on configuration (excluding cache-related flags for special handling)
        cache_related_keys = {"cached_nodes", "cache_nodes", "cached_nodes_count", "use_dynamic_cache", "cache_file"}
        
        for config_key, arg_name in arg_mapping.items():
            if config_key in config and config_key not in cache_related_keys:
                value = config[config_key]
                
                # Handle boolean flags
                if isinstance(value, bool):
                    if value:
                        args.append(arg_name)
                else:
                    args.extend([arg_name, str(value)])
        
        # Special handling for cache flags
        # If using cached nodes, don't add --cache-nodes flag (don't regenerate cache)
        if config.get("cached_nodes", False):
            # Add --use-cached flag (boolean flag, no value)
            args.append("--use-cached")
            
            # Check if we should use dynamic cache detection
            if config.get("use_dynamic_cache", False):
                # Add a flag to indicate dynamic cache detection
                args.append("--dynamic-cache-detection")
            else:
                # Use the specified cached_nodes_count
                if "cached_nodes_count" in config:
                    args.extend(["--cached-nodes", str(config["cached_nodes_count"])])
        else:
            # If not using cached nodes, check if we should cache nodes
            if config.get("cache_nodes", False):
                args.append("--cache-nodes")
                # Add cached_nodes_count if specified
                if "cached_nodes_count" in config:
                    args.extend(["--cached-nodes", str(config["cached_nodes_count"])])
        
        # Always include the cache file argument
        cache_file = config.get("cache_file", "node_cache/cached_nodes.pkl")
        args.extend(["--cache-file", cache_file])
        
        # Debug output for cache-related issues
        print(f"DEBUG: Cache configuration:")
        print(f"  cached_nodes: {config.get('cached_nodes', False)}")
        print(f"  cache_nodes: {config.get('cache_nodes', False)}")
        print(f"  use_dynamic_cache: {config.get('use_dynamic_cache', False)}")
        print(f"  cached_nodes_count: {config.get('cached_nodes_count', 'Not set')}")
        print(f"  cache_file: {cache_file}")
        print(f"  Final command args: {args}")
        
        # Always append --run-id <run_id> if available
        if run_id:
            args.extend(["--run-id", run_id])
        return args
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all test runs."""
        runs = []
        for run_file in self.runs_dir.glob("*.json"):
            if run_file.stem.startswith("run_"):
                metadata = self._load_run_metadata(run_file.stem)
                if metadata:
                    # Add summary information
                    summary = {
                        "run_id": metadata.get("run_id", run_file.stem),
                        "config_name": metadata.get("config_name", "Unknown"),
                        "status": metadata.get("status", "Unknown"),
                        "start_time": metadata.get("start_time", "Unknown"),
                        "end_time": metadata.get("end_time", None),
                        "duration": self._calculate_duration(metadata),
                        "final_accuracy": metadata.get("results", {}).get("final_accuracy", None)
                    }
                    runs.append(summary)
        
        return sorted(runs, key=lambda x: x["start_time"], reverse=True)
    
    def get_active_runs(self) -> List[Dict[str, Any]]:
        """Get list of currently running tests."""
        active = []
        for run_id, process in self.active_processes.items():
            if process.poll() is None:  # Still running
                metadata = self._load_run_metadata(run_id)
                if metadata:
                    active.append({
                        "run_id": run_id,
                        "config_name": metadata.get("config_name", "Unknown"),
                        "start_time": metadata.get("start_time", "Unknown"),
                        "pid": process.pid
                    })
        return active
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific test run."""
        metadata = self._load_run_metadata(run_id)
        if not metadata:
            return None
        
        # Optionally add some recent log lines
        metadata["logs_preview"] = self.get_run_logs(run_id, tail_lines=20)
        
        # Calculate duration if not present
        if "duration" not in metadata:
            metadata["duration"] = self._calculate_duration(metadata)
        
        return metadata
    
    def get_run_logs(self, run_id: str, tail_lines: int = 0) -> List[str]:
        """
        Get logs for a specific test run, processing carriage returns to handle
        progress bars correctly.
        """
        try:
            metadata = self._load_run_metadata(run_id)
            if not metadata or "log_file" not in metadata:
                return ["Log file not found."]
            
            log_file = Path(metadata["log_file"])
            if not log_file.exists():
                return ["Log file does not exist."]
            
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            processed_lines = []
            for line in lines:
                # Get the part of the line after the last carriage return
                # This handles TQDM's single-line updates
                line_content = line.strip().rsplit('\r', 1)[-1]
                
                if not line_content:
                    continue
                
                # Enhanced heuristic to detect TQDM progress bar lines
                # Look for patterns like: "Basic Training Epoch N/A: 3%|3 | 1/32 [00:01<00:52, 1.70s/batch]"
                is_progress_line = (
                    # Check for percentage and progress bar patterns
                    ('%|' in line_content and '|' in line_content) or
                    # Check for time-based patterns (s/batch, batch/s, it/s)
                    any(pattern in line_content for pattern in ['s/batch', 'batch/s', 'it/s']) or
                    # Check for ETA patterns
                    ('ETA' in line_content and ('<' in line_content or '>' in line_content)) or
                    # Check for progress bar with hash symbols
                    ('%|#' in line_content and '|' in line_content)
                )

                # If current line is a progress bar and the last processed line was also one,
                # replace the last one to create an "in-place" update effect.
                if is_progress_line and processed_lines:
                    last_line = processed_lines[-1]
                    # Use the same detection logic for the last line
                    is_last_line_progress = (
                        ('%|' in last_line and '|' in last_line) or
                        any(pattern in last_line for pattern in ['s/batch', 'batch/s', 'it/s']) or
                        ('ETA' in last_line and ('<' in last_line or '>' in last_line)) or
                        ('%|#' in last_line and '|' in last_line)
                    )
                    if is_last_line_progress:
                        processed_lines[-1] = line_content
                        continue

                processed_lines.append(line_content)

            if tail_lines > 0:
                return processed_lines[-tail_lines:]
            else:
                return processed_lines

        except Exception as e:
            print(f"Error reading log file for run {run_id}: {e}")
            return [f"Error reading logs: {e}"]
    
    def stop_run(self, run_id: str) -> bool:
        """Stop a running test."""
        try:
            print(f"\nAttempting to stop run {run_id}")
            if run_id not in self.active_processes:
                print(f"Run {run_id} not found in active processes")
                return False
            
            process = self.active_processes[run_id]
            print(f"Found process with PID {process.pid}")
            
            # Try multiple methods to stop the process
            try:
                # Method 1: Try process group termination
                try:
                    pgid = os.getpgid(process.pid)
                    print(f"Process group ID: {pgid}")
                    print(f"Sending SIGTERM to process group {pgid}")
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError) as e:
                    print(f"Process group termination failed: {e}")
                
                # Method 2: Try direct process termination
                try:
                    print(f"Sending SIGTERM to process {process.pid}")
                    process.terminate()
                except Exception as e:
                    print(f"Direct process termination failed: {e}")
                
                # Method 3: Try psutil for process and children
                try:
                    print("Attempting psutil process termination...")
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                    print(f"Found {len(children)} child processes")
                    
                    # Terminate children first
                    for child in children:
                        try:
                            print(f"Terminating child process {child.pid}")
                            child.terminate()
                        except psutil.NoSuchProcess:
                            print(f"Child process {child.pid} already terminated")
                        except Exception as e:
                            print(f"Error terminating child {child.pid}: {e}")
                    
                    # Terminate parent
                    try:
                        print(f"Terminating parent process {parent.pid}")
                        parent.terminate()
                    except psutil.NoSuchProcess:
                        print(f"Parent process {parent.pid} already terminated")
                    except Exception as e:
                        print(f"Error terminating parent: {e}")
                except Exception as e:
                    print(f"Psutil termination failed: {e}")
                
                # Wait for processes to terminate
                print("Waiting for processes to terminate...")
                time.sleep(2)
                
                # Check if process is still running
                if process.poll() is None:
                    print("Process still running, attempting force kill...")
                    
                    # Method 4: Force kill with SIGKILL
                    try:
                        # Try process group first
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
                        
                        # Try direct process kill
                        try:
                            process.kill()
                        except Exception as e:
                            print(f"Direct process kill failed: {e}")
                        
                        # Try psutil force kill
                        try:
                            if parent.is_running():
                                parent.kill()
                            for child in children:
                                if child.is_running():
                                    child.kill()
                        except Exception as e:
                            print(f"Psutil force kill failed: {e}")
                    except Exception as e:
                        print(f"Force kill failed: {e}")
                
                # Update status
                print("Updating run status to stopped")
                self._update_run_status(run_id, "stopped", 
                                      end_time=datetime.now().isoformat())
                return True
                
            except ProcessLookupError as e:
                print(f"Process already terminated: {e}")
                self._update_run_status(run_id, "stopped",
                                      end_time=datetime.now().isoformat())
                return True
            except Exception as e:
                print(f"Error during process termination: {e}")
                raise
        
        except Exception as e:
            print(f"Error stopping run {run_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare results from multiple runs."""
        comparison = {
            "runs": [],
            "metrics": {},
            "summary": {}
        }
        
        for run_id in run_ids:
            metadata = self._load_run_metadata(run_id)
            if metadata:
                run_summary = {
                    "run_id": run_id,
                    "config_name": metadata.get("config_name", "Unknown"),
                    "status": metadata.get("status", "Unknown"),
                    "duration": self._calculate_duration(metadata),
                    "config": metadata.get("config", {}),
                    "results": metadata.get("results", {})
                }
                comparison["runs"].append(run_summary)
        
        # Extract metrics for comparison
        accuracy_values = []
        for run in comparison["runs"]:
            acc = run["results"].get("final_accuracy")
            if acc is not None:
                accuracy_values.append(acc)
        
        if accuracy_values:
            comparison["metrics"]["accuracy"] = {
                "values": accuracy_values,
                "mean": sum(accuracy_values) / len(accuracy_values),
                "min": min(accuracy_values),
                "max": max(accuracy_values),
                "std": None  # Could calculate if needed
            }
        
        # Generate summary
        comparison["summary"]["total_runs"] = len(comparison["runs"])
        comparison["summary"]["completed_runs"] = len([r for r in comparison["runs"] if r["status"] == "completed"])
        comparison["summary"]["best_accuracy"] = max(accuracy_values) if accuracy_values else None
        
        return comparison
    
    # Helper methods
    def _save_run_metadata(self, run_id: str, metadata: Dict[str, Any]) -> bool:
        """Save run metadata to file."""
        try:
            run_file = self.runs_dir / f"{run_id}.json"
            with open(run_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving run metadata for {run_id}: {e}")
            return False
    
    def _load_run_metadata(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load run metadata from file."""
        try:
            run_file = self.runs_dir / f"{run_id}.json"
            if not run_file.exists():
                return None
            with open(run_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading run metadata for {run_id}: {e}")
            return None
    
    def _update_run_status(self, run_id: str, status: str, **kwargs) -> bool:
        """Update run status and additional fields."""
        metadata = self._load_run_metadata(run_id)
        if not metadata:
            return False
        
        metadata["status"] = status
        metadata["last_updated"] = datetime.now().isoformat()
        
        # Update additional fields
        for key, value in kwargs.items():
            metadata[key] = value
        
        return self._save_run_metadata(run_id, metadata)
    
    def _monitor_process(self, run_id: str, process: subprocess.Popen, log_file: str):
        """Monitor a test process in a separate thread."""
        try:
            # Wait for process to complete
            return_code = process.wait()
            
            # Update status based on return code
            if return_code == 0:
                self._update_run_status(run_id, "completed", end_time=datetime.now().isoformat())
                self._extract_results(run_id)
            else:
                self._update_run_status(run_id, "failed", 
                                      end_time=datetime.now().isoformat(),
                                      error=f"Process exited with code {return_code}")
        
        except Exception as e:
            self._update_run_status(run_id, "failed",
                                  end_time=datetime.now().isoformat(),
                                  error=str(e))
        
        finally:
            # Clean up tracking
            if run_id in self.active_processes:
                del self.active_processes[run_id]
            if run_id in self.process_threads:
                del self.process_threads[run_id]
    
    def _extract_results(self, run_id: str):
        """Extract results from completed test run."""
        try:
            metadata = self._load_run_metadata(run_id)
            if not metadata:
                return
            
            # Look for result files in expected locations
            config_name = metadata.get("config_name", "unknown")
            
            # Try to find bias visualization files
            bias_viz_dir = Path(f"bias_visualizations/{config_name}")
            if bias_viz_dir.exists():
                bias_files = list(bias_viz_dir.glob("*.json"))
                if bias_files:
                    # Load the most recent bias metrics
                    latest_bias_file = max(bias_files, key=lambda x: x.stat().st_mtime)
                    try:
                        with open(latest_bias_file, 'r') as f:
                            bias_data = json.load(f)
                        metadata["results"] = {
                            "bias_metrics": bias_data,
                            "bias_plots": [str(f) for f in bias_viz_dir.glob("*.png")]
                        }
                    except Exception as e:
                        print(f"Error loading bias results for {run_id}: {e}")
            
            # Try to extract final accuracy from logs
            log_file = metadata.get("log_file")
            if log_file and os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    
                    # Look for final test results
                    if "Final Test Results" in log_content:
                        # Extract accuracy from log (simplified extraction)
                        lines = log_content.split('\n')
                        for i, line in enumerate(lines):
                            if "Final Test Results" in line and i + 1 < len(lines):
                                # Try to parse accuracy from next few lines
                                for j in range(i + 1, min(i + 10, len(lines))):
                                    if '"accuracy"' in lines[j]:
                                        try:
                                            # Attempt to extract accuracy value
                                            import re
                                            acc_match = re.search(r'"accuracy":\s*([0-9.]+)', lines[j])
                                            if acc_match:
                                                if "results" not in metadata:
                                                    metadata["results"] = {}
                                                metadata["results"]["final_accuracy"] = float(acc_match.group(1))
                                                break
                                        except:
                                            pass
                except Exception as e:
                    print(f"Error extracting results from log for {run_id}: {e}")
            
            # Save updated metadata
            self._save_run_metadata(run_id, metadata)
            
        except Exception as e:
            print(f"Error extracting results for {run_id}: {e}")
    
    def _calculate_duration(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Calculate run duration."""
        try:
            start_time = metadata.get("start_time")
            end_time = metadata.get("end_time")
            
            if not start_time:
                return None
            
            start_dt = datetime.fromisoformat(start_time)
            
            if end_time:
                end_dt = datetime.fromisoformat(end_time)
            else:
                end_dt = datetime.now()
            
            duration = end_dt - start_dt
            
            # Format duration
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        
        except Exception:
            return None 