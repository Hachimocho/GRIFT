import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import json
import os
from datetime import datetime
import pandas as pd
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

class IValueVisualizationTracker:
    """
    Tracks and visualizes I-value changes during model training.
    Provides multiple visualization strategies that scale to large datasets.
    """
    
    def __init__(self, save_dir="ivalue_visualizations", max_history_length=1000):
        """
        Initialize the I-value visualization tracker.
        
        Args:
            save_dir: Directory to save visualization plots and data
            max_history_length: Maximum number of time points to store in memory
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Time series data storage
        self.epoch_stats = []  # Overall statistics per epoch
        self.step_stats = deque(maxlen=max_history_length)  # Step-level statistics
        self.subgroup_stats = defaultdict(list)  # Statistics by attribute subgroups
        
        # Sample tracking for specific nodes
        self.tracked_nodes = {}  # node_id -> node_data for detailed tracking
        self.node_history = defaultdict(list)  # node_id -> [(epoch, step, i_value)]
        
        # Distribution tracking
        self.distribution_snapshots = []  # Full I-value distributions at key moments
        
        # Bias hop tracking (for cluster hop traversal)
        self.bias_hop_history = []
        
        self.current_epoch = 0
        self.current_step = 0
        
    def start_epoch(self, epoch):
        """Mark the start of a new training epoch."""
        self.current_epoch = max(0, int(epoch))  # Ensure non-negative integer epoch
        self.current_step = 0
        
    def log_step_statistics(self, trainer, nodes_batch=None, traversal=None):
        """
        Log I-value statistics for the current training step.
        
        Args:
            trainer: The trainer instance (IValueTrainer or AdaptiveTrainer)
            nodes_batch: Optional batch of nodes to analyze
            traversal: Optional traversal instance for additional data
        """
        self.current_step += 1
        
        stats = {
            'epoch': max(0, int(self.current_epoch)),  # Ensure non-negative integer
            'step': self.current_step,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get I-values for analysis
        if nodes_batch:
            i_values = self._collect_i_values(trainer, nodes_batch)
            if i_values:
                stats.update(self._calculate_aggregate_stats(i_values))
                stats['batch_size'] = len(nodes_batch)
        
        # Track subgroup statistics if attribute metadata is available
        if hasattr(trainer, 'attribute_metadata') and trainer.attribute_metadata and nodes_batch:
            subgroup_stats = self._calculate_subgroup_stats(trainer, nodes_batch)
            stats['subgroup_stats'] = subgroup_stats
            
        # Track bias hop information for cluster hop traversal
        if traversal and hasattr(traversal, 'get_hop_i_value_history'):
            hop_history = traversal.get_hop_i_value_history()
            if hop_history:
                stats['bias_hops'] = hop_history[-1] if hop_history else None
                
        self.step_stats.append(stats)
        
    def log_epoch_summary(self, trainer, sample_nodes=None, sample_size=1000):
        """
        Log comprehensive I-value statistics at the end of an epoch.
        
        Args:
            trainer: The trainer instance
            sample_nodes: Optional nodes to sample from (if None, samples from graph)
            sample_size: Number of nodes to sample for analysis
        """
        # Sample nodes if not provided
        if sample_nodes is None and hasattr(trainer, 'graphmanager'):
            all_nodes = list(trainer.graphmanager.get_graph().get_nodes())
            sample_nodes = random.sample(all_nodes, min(sample_size, len(all_nodes)))
        
        if not sample_nodes:
            return
            
        i_values = self._collect_i_values(trainer, sample_nodes)
        if not i_values:
            return
            
        epoch_stats = {
            'epoch': max(0, int(self.current_epoch)),  # Ensure non-negative integer
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(sample_nodes)
        }
        
        # Calculate comprehensive statistics
        epoch_stats.update(self._calculate_aggregate_stats(i_values))
        
        # Calculate subgroup statistics
        if hasattr(trainer, 'attribute_metadata') and trainer.attribute_metadata:
            subgroup_stats = self._calculate_subgroup_stats(trainer, sample_nodes)
            epoch_stats['subgroup_stats'] = subgroup_stats
            
        # Store distribution snapshot
        if self.current_epoch % 5 == 0:  # Every 5 epochs
            epoch_stats['i_value_distribution'] = i_values
            self.distribution_snapshots.append({
                'epoch': max(0, int(self.current_epoch)),
                'i_values': i_values
            })
            
        self.epoch_stats.append(epoch_stats)
        
    def track_specific_nodes(self, trainer, node_ids_or_nodes, max_nodes=50):
        """
        Set up tracking for specific nodes throughout training.
        
        Args:
            trainer: The trainer instance
            node_ids_or_nodes: List of node IDs or node objects to track
            max_nodes: Maximum number of nodes to track
        """
        nodes_to_track = node_ids_or_nodes[:max_nodes]
        
        for node_item in nodes_to_track:
            if hasattr(node_item, 'node_id'):  # It's a node object
                node_id = node_item.node_id
                self.tracked_nodes[node_id] = node_item
                # Initialize with empty list for this node
                if node_id not in self.node_history:
                    self.node_history[node_id] = []
            else:  # It's a node ID
                node_id = node_item
                # We'll need to find the node object later
                if node_id not in self.node_history:
                    self.node_history[node_id] = []
                
        print(f"Tracking {len(nodes_to_track)} specific nodes for detailed I-value analysis")
        
    def update_tracked_nodes(self, trainer):
        """Update I-values for all tracked nodes at the current epoch."""
        for node_id, node in self.tracked_nodes.items():
            try:
                i_value = trainer.get_i_value(node, 0)
                # Store with integer epoch to avoid decimal/negative epochs
                self.node_history[node_id].append((
                    max(0, int(self.current_epoch)),  # Ensure non-negative integer epoch
                    self.current_step,
                    i_value
                ))
            except Exception as e:
                print(f"Error updating tracked node {node_id}: {e}")
                
    def _collect_i_values(self, trainer, nodes):
        """Collect I-values for a set of nodes."""
        i_values = []
        for node in nodes:
            try:
                i_value = trainer.get_i_value(node, 0)
                if isinstance(i_value, (int, float)) and not np.isnan(i_value):
                    i_values.append(i_value)
            except Exception as e:
                continue  # Skip problematic nodes
        return i_values
        
    def _calculate_aggregate_stats(self, i_values):
        """Calculate aggregate statistics for I-values."""
        if not i_values:
            return {}
            
        i_values = np.array(i_values)
        return {
            'mean': float(np.mean(i_values)),
            'median': float(np.median(i_values)),
            'std': float(np.std(i_values)),
            'min': float(np.min(i_values)),
            'max': float(np.max(i_values)),
            'q25': float(np.percentile(i_values, 25)),
            'q75': float(np.percentile(i_values, 75)),
            'high_i_value_ratio': float(np.mean(i_values > 0.7)),  # Ratio of high I-values
            'low_i_value_ratio': float(np.mean(i_values < 0.3)),   # Ratio of low I-values
        }
        
    def _calculate_subgroup_stats(self, trainer, nodes):
        """Calculate I-value statistics by attribute subgroups."""
        if not hasattr(trainer, 'attribute_metadata') or not trainer.attribute_metadata:
            return {}
            
        subgroup_data = defaultdict(list)
        categorical_attrs = [
            attr for attr in trainer.attribute_metadata 
            if (isinstance(attr, dict) and attr.get('type') == 'categorical') or
               (hasattr(attr, 'attr_type') and attr.attr_type == 'categorical')
        ]
        
        for node in nodes:
            if not hasattr(node, 'attributes') or not node.attributes:
                continue
                
            try:
                i_value = trainer.get_i_value(node, 0)
                if not isinstance(i_value, (int, float)) or np.isnan(i_value):
                    continue
                    
                # Create subgroup keys
                for attr in categorical_attrs:
                    attr_name = attr['name'] if isinstance(attr, dict) else attr.name
                    if attr_name in node.attributes:
                        attr_value = node.attributes[attr_name]
                        subgroup_key = f"{attr_name}_{attr_value}"
                        subgroup_data[subgroup_key].append(i_value)
                        
            except Exception as e:
                continue
                
        # Calculate statistics for each subgroup
        subgroup_stats = {}
        for subgroup, values in subgroup_data.items():
            if values:
                subgroup_stats[subgroup] = self._calculate_aggregate_stats(values)
                subgroup_stats[subgroup]['count'] = len(values)
                
        return subgroup_stats
        
    def plot_training_progression(self, save_path=None):
        """Plot I-value statistics progression over training."""
        if not self.epoch_stats:
            print("No epoch statistics to plot")
            return
            
        # Ensure epochs are integers and non-negative
        epochs = [max(0, int(stat['epoch'])) for stat in self.epoch_stats]
        means = [stat.get('mean', 0) for stat in self.epoch_stats]
        stds = [stat.get('std', 0) for stat in self.epoch_stats]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mean I-value over time
        axes[0, 0].plot(epochs, means, 'b-', linewidth=2)
        axes[0, 0].fill_between(epochs, 
                               [m - s for m, s in zip(means, stds)],
                               [m + s for m, s in zip(means, stds)],
                               alpha=0.3)
        axes[0, 0].set_title('Mean I-value ± Std Dev Over Training')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('I-value')
        axes[0, 0].grid(True)
        
        # Set integer x-axis ticks
        if epochs:
            axes[0, 0].set_xticks(range(min(epochs), max(epochs) + 1))
        
        # I-value distribution evolution
        if len(epochs) > 1:
            mins = [stat.get('min', 0) for stat in self.epoch_stats]
            maxs = [stat.get('max', 1) for stat in self.epoch_stats]
            q25s = [stat.get('q25', 0) for stat in self.epoch_stats]
            q75s = [stat.get('q75', 1) for stat in self.epoch_stats]
            
            axes[0, 1].plot(epochs, q25s, 'g--', label='Q25', alpha=0.7)
            axes[0, 1].plot(epochs, means, 'b-', label='Mean', linewidth=2)
            axes[0, 1].plot(epochs, q75s, 'r--', label='Q75', alpha=0.7)
            axes[0, 1].fill_between(epochs, mins, maxs, alpha=0.2, label='Min-Max Range')
            axes[0, 1].set_title('I-value Distribution Evolution')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('I-value')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Set integer x-axis ticks
            if epochs:
                axes[0, 1].set_xticks(range(min(epochs), max(epochs) + 1))
        
        # High/Low I-value ratios
        high_ratios = [stat.get('high_i_value_ratio', 0) for stat in self.epoch_stats]
        low_ratios = [stat.get('low_i_value_ratio', 0) for stat in self.epoch_stats]
        
        axes[1, 0].plot(epochs, high_ratios, 'r-', label='High I-value (>0.7)', linewidth=2)
        axes[1, 0].plot(epochs, low_ratios, 'b-', label='Low I-value (<0.3)', linewidth=2)
        axes[1, 0].set_title('Proportion of High/Low I-values')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Proportion')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Set integer x-axis ticks
        if epochs:
            axes[1, 0].set_xticks(range(min(epochs), max(epochs) + 1))
        
        # Sample sizes
        sample_sizes = [stat.get('sample_size', 0) for stat in self.epoch_stats]
        axes[1, 1].bar(epochs, sample_sizes, alpha=0.7)
        axes[1, 1].set_title('Sample Sizes per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Number of Nodes')
        axes[1, 1].grid(True)
        
        # Set integer x-axis ticks
        if epochs:
            axes[1, 1].set_xticks(range(min(epochs), max(epochs) + 1))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / f"training_progression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training progression plot saved to: {save_path}")
        
    def plot_subgroup_analysis(self, save_path=None):
        """Plot I-value statistics by attribute subgroups."""
        if not self.epoch_stats or not any(stat.get('subgroup_stats') for stat in self.epoch_stats):
            print("No subgroup statistics to plot")
            return
            
        # Collect all subgroups across epochs
        all_subgroups = set()
        for stat in self.epoch_stats:
            if 'subgroup_stats' in stat:
                all_subgroups.update(stat['subgroup_stats'].keys())
                
        if not all_subgroups:
            return
            
        # Create subgroup progression plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = [stat['epoch'] for stat in self.epoch_stats]
        
        # Mean I-values by subgroup
        for subgroup in list(all_subgroups)[:8]:  # Limit to avoid clutter
            means = []
            for stat in self.epoch_stats:
                subgroup_data = stat.get('subgroup_stats', {}).get(subgroup, {})
                means.append(subgroup_data.get('mean', np.nan))
            
            # Only plot if we have valid data
            if not all(np.isnan(means)):
                axes[0, 0].plot(epochs, means, label=subgroup[:20], alpha=0.7)
        
        axes[0, 0].set_title('Mean I-values by Subgroup')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Mean I-value')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True)
        
        # Subgroup variance over time
        for subgroup in list(all_subgroups)[:8]:
            stds = []
            for stat in self.epoch_stats:
                subgroup_data = stat.get('subgroup_stats', {}).get(subgroup, {})
                stds.append(subgroup_data.get('std', np.nan))
            
            if not all(np.isnan(stds)):
                axes[0, 1].plot(epochs, stds, label=subgroup[:20], alpha=0.7)
        
        axes[0, 1].set_title('I-value Std Dev by Subgroup')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Std Dev')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True)
        
        # Latest epoch subgroup comparison
        if self.epoch_stats:
            latest_stats = self.epoch_stats[-1].get('subgroup_stats', {})
            if latest_stats:
                subgroups = list(latest_stats.keys())
                means = [latest_stats[sg].get('mean', 0) for sg in subgroups]
                
                axes[1, 0].bar(range(len(subgroups)), means, alpha=0.7)
                axes[1, 0].set_title(f'Latest Epoch ({epochs[-1]}) Mean I-values by Subgroup')
                axes[1, 0].set_xlabel('Subgroup')
                axes[1, 0].set_ylabel('Mean I-value')
                axes[1, 0].set_xticks(range(len(subgroups)))
                axes[1, 0].set_xticklabels([sg[:15] for sg in subgroups], rotation=45, ha='right')
                axes[1, 0].grid(True)
        
        # Subgroup sample sizes
        if self.epoch_stats:
            latest_stats = self.epoch_stats[-1].get('subgroup_stats', {})
            if latest_stats:
                subgroups = list(latest_stats.keys())
                counts = [latest_stats[sg].get('count', 0) for sg in subgroups]
                
                axes[1, 1].bar(range(len(subgroups)), counts, alpha=0.7)
                axes[1, 1].set_title(f'Latest Epoch ({epochs[-1]}) Sample Sizes by Subgroup')
                axes[1, 1].set_xlabel('Subgroup')
                axes[1, 1].set_ylabel('Sample Count')
                axes[1, 1].set_xticks(range(len(subgroups)))
                axes[1, 1].set_xticklabels([sg[:15] for sg in subgroups], rotation=45, ha='right')
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / f"subgroup_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Subgroup analysis plot saved to: {save_path}")
        
    def plot_tracked_nodes(self, save_path=None):
        """Plot I-value evolution for specific tracked nodes."""
        if not self.node_history:
            print("No tracked nodes to plot")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Individual node trajectories
        for node_id, history in list(self.node_history.items())[:10]:  # Limit to 10 nodes
            if history:
                # Ensure epochs are integers and non-negative
                epochs = [max(0, int(h[0])) for h in history]
                i_values = [h[2] for h in history]
                axes[0].plot(epochs, i_values, label=f'Node {node_id}', alpha=0.7, marker='o', markersize=3)
        
        axes[0].set_title('I-value Evolution for Tracked Nodes')
        axes[0].set_xlabel('Training Epoch')
        axes[0].set_ylabel('I-value')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True)
        
        # Set integer x-axis ticks if we have data
        all_epochs = set()
        for history in self.node_history.values():
            all_epochs.update(max(0, int(h[0])) for h in history)
        if all_epochs:
            axes[0].set_xticks(range(min(all_epochs), max(all_epochs) + 1))
        
        # Average trajectory across all tracked nodes
        if self.node_history:
            all_epochs = set()
            for history in self.node_history.values():
                all_epochs.update(max(0, int(h[0])) for h in history)
            
            all_epochs = sorted(all_epochs)
            avg_i_values = []
            std_i_values = []
            
            for epoch in all_epochs:
                epoch_i_values = []
                for history in self.node_history.values():
                    # Find i-value for this epoch
                    epoch_data = [h for h in history if max(0, int(h[0])) == epoch]
                    if epoch_data:
                        epoch_i_values.append(epoch_data[-1][2])  # Take the latest from this epoch
                
                if epoch_i_values:
                    avg_i_values.append(np.mean(epoch_i_values))
                    std_i_values.append(np.std(epoch_i_values))
                else:
                    avg_i_values.append(np.nan)
                    std_i_values.append(np.nan)
            
            # Plot average with error bands
            axes[1].plot(all_epochs, avg_i_values, 'b-', linewidth=3, marker='o', markersize=4, label='Average')
            axes[1].fill_between(all_epochs,
                               [v - s if not np.isnan(v) and not np.isnan(s) else v for v, s in zip(avg_i_values, std_i_values)],
                               [v + s if not np.isnan(v) and not np.isnan(s) else v for v, s in zip(avg_i_values, std_i_values)],
                               alpha=0.3, label='± Std Dev')
            
        axes[1].set_title('Average I-value Evolution (Tracked Nodes)')
        axes[1].set_xlabel('Training Epoch')
        axes[1].set_ylabel('I-value')
        axes[1].legend()
        axes[1].grid(True)
        
        # Set integer x-axis ticks
        if all_epochs:
            axes[1].set_xticks(range(min(all_epochs), max(all_epochs) + 1))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / f"tracked_nodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Tracked nodes plot saved to: {save_path}")
        
    def save_data(self, filename=None):
        """Save all collected data to JSON file."""
        if filename is None:
            filename = self.save_dir / f"ivalue_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        # Convert complex data structures to JSON-serializable format
        def make_json_serializable(obj):
            """Convert objects to JSON-serializable format."""
            if isinstance(obj, (tuple, list)):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): make_json_serializable(v) for k, v in obj.items()}
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        # Prepare data with proper serialization
        data = {
            'epoch_stats': make_json_serializable(self.epoch_stats),
            'step_stats': make_json_serializable(list(self.step_stats)),
            'node_history': {
                str(k): make_json_serializable(v) 
                for k, v in self.node_history.items()
            },
            'distribution_snapshots': make_json_serializable(self.distribution_snapshots),
            'bias_hop_history': make_json_serializable(self.bias_hop_history)
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"I-value data saved to: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to save I-value data as JSON: {e}")
            # Try saving as pickle as fallback
            try:
                import pickle
                pickle_filename = str(filename).replace('.json', '.pkl')
                with open(pickle_filename, 'wb') as f:
                    pickle.dump({
                        'epoch_stats': self.epoch_stats,
                        'step_stats': list(self.step_stats),
                        'node_history': dict(self.node_history),
                        'distribution_snapshots': self.distribution_snapshots,
                        'bias_hop_history': self.bias_hop_history
                    }, f)
                print(f"I-value data saved as pickle to: {pickle_filename}")
            except Exception as pickle_error:
                print(f"Failed to save data in any format: {pickle_error}")
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        if not self.epoch_stats:
            print("No data available for summary report")
            return
            
        print("\n" + "="*60)
        print("I-VALUE TRAINING SUMMARY REPORT")
        print("="*60)
        
        # Training overview
        print(f"Total epochs: {len(self.epoch_stats)}")
        print(f"Total steps tracked: {len(self.step_stats)}")
        print(f"Nodes tracked individually: {len(self.node_history)}")
        
        # Latest statistics
        if self.epoch_stats:
            latest = self.epoch_stats[-1]
            print(f"\nLatest Epoch ({latest['epoch']}):")
            print(f"  Mean I-value: {latest.get('mean', 'N/A'):.4f}")
            print(f"  Std Dev: {latest.get('std', 'N/A'):.4f}")
            print(f"  Min: {latest.get('min', 'N/A'):.4f}")
            print(f"  Max: {latest.get('max', 'N/A'):.4f}")
            print(f"  High I-value ratio (>0.7): {latest.get('high_i_value_ratio', 'N/A'):.2%}")
            print(f"  Low I-value ratio (<0.3): {latest.get('low_i_value_ratio', 'N/A'):.2%}")
        
        # Trends
        if len(self.epoch_stats) >= 2:
            first_mean = self.epoch_stats[0].get('mean', 0)
            last_mean = self.epoch_stats[-1].get('mean', 0)
            mean_change = last_mean - first_mean
            
            print(f"\nTrends:")
            print(f"  Mean I-value change: {mean_change:+.4f}")
            print(f"  Direction: {'Increasing' if mean_change > 0 else 'Decreasing' if mean_change < 0 else 'Stable'}")
        
        print("="*60) 