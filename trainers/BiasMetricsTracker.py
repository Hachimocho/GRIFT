import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class BiasMetricsTracker:
    """
    Tracks and visualizes bias metrics during model training.
    Monitors race-gender subgroup fairness, overall bias trends, and attribute-level bias.
    """
    
    def __init__(self, save_dir="bias_visualizations"):
        """
        Initialize the bias metrics tracker.
        
        Args:
            save_dir: Directory to save bias visualization plots and data
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Bias metrics storage by epoch
        self.train_bias_history = []
        self.val_bias_history = []
        self.test_bias_history = []
        
        # Overall metrics tracking
        self.overall_metrics_history = []
        
        print(f"🎯 BiasMetricsTracker initialized - saving to: {self.save_dir}")
    
    def log_bias_metrics(self, epoch, train_metrics=None, val_metrics=None, test_metrics=None):
        """
        Log bias metrics for a given epoch.
        
        Args:
            epoch: Training epoch number (should be non-negative integer)
            train_metrics: Training bias metrics from evaluate_model()
            val_metrics: Validation bias metrics from evaluate_model()  
            test_metrics: Test bias metrics from evaluate_model()
        """
        timestamp = datetime.now().isoformat()
        epoch = max(0, int(epoch))  # Ensure non-negative integer epoch
        
        # Store individual split metrics
        if train_metrics and 'bias_metrics' in train_metrics:
            bias_data = train_metrics['bias_metrics'].copy()
            bias_data.update({
                'epoch': epoch,
                'timestamp': timestamp,
                'split': 'train',
                'overall_accuracy': train_metrics.get('accuracy', 0.0) / 100.0  # Convert to decimal
            })
            self.train_bias_history.append(bias_data)
        
        if val_metrics and 'bias_metrics' in val_metrics:
            bias_data = val_metrics['bias_metrics'].copy()
            bias_data.update({
                'epoch': epoch,
                'timestamp': timestamp,
                'split': 'val',
                'overall_accuracy': val_metrics.get('accuracy', 0.0) / 100.0
            })
            self.val_bias_history.append(bias_data)
        
        if test_metrics and 'bias_metrics' in test_metrics:
            bias_data = test_metrics['bias_metrics'].copy()
            bias_data.update({
                'epoch': epoch,
                'timestamp': timestamp,
                'split': 'test',
                'overall_accuracy': test_metrics.get('accuracy', 0.0) / 100.0
            })
            self.test_bias_history.append(bias_data)
        
        # Aggregate overall metrics
        overall_data = {
            'epoch': epoch,
            'timestamp': timestamp,
            'train_accuracy': train_metrics.get('accuracy', None) if train_metrics else None,
            'val_accuracy': val_metrics.get('accuracy', None) if val_metrics else None,
            'test_accuracy': test_metrics.get('accuracy', None) if test_metrics else None,
            'train_bias': train_metrics.get('bias_metrics', {}).get('race_gender_overall_bias', None) if train_metrics else None,
            'val_bias': val_metrics.get('bias_metrics', {}).get('race_gender_overall_bias', None) if val_metrics else None,
            'test_bias': test_metrics.get('bias_metrics', {}).get('race_gender_overall_bias', None) if test_metrics else None
        }
        self.overall_metrics_history.append(overall_data)
        
        print(f"📊 Logged bias metrics for epoch {epoch}")
    
    def plot_bias_evolution(self, save_path=None):
        """Plot bias metrics evolution over training epochs."""
        if not any([self.train_bias_history, self.val_bias_history, self.test_bias_history]):
            print("No bias metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Bias Metrics Evolution During Training', fontsize=16, fontweight='bold')
        
        # Overall bias trend (max accuracy difference across subgroups)
        for history, label, color in [
            (self.train_bias_history, 'Train', 'blue'),
            (self.val_bias_history, 'Validation', 'orange'), 
            (self.test_bias_history, 'Test', 'green')
        ]:
            if history:
                epochs = [h['epoch'] for h in history]
                overall_bias = [h.get('race_gender_overall_bias', 0) for h in history]
                axes[0, 0].plot(epochs, overall_bias, 'o-', label=label, color=color, linewidth=2, markersize=4)
        
        axes[0, 0].set_title('Race-Gender Overall Bias (Max Accuracy Difference)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Bias (Max Acc Diff)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average subgroup bias trend
        for history, label, color in [
            (self.train_bias_history, 'Train', 'blue'),
            (self.val_bias_history, 'Validation', 'orange'),
            (self.test_bias_history, 'Test', 'green')
        ]:
            if history:
                epochs = [h['epoch'] for h in history]
                avg_bias = [h.get('race_gender_average_subgroup_bias', 0) for h in history]
                axes[0, 1].plot(epochs, avg_bias, 'o-', label=label, color=color, linewidth=2, markersize=4)
        
        axes[0, 1].set_title('Race-Gender Average Subgroup Bias')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Avg Subgroup Bias')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Per-attribute bias evolution
        for history, label, color in [
            (self.train_bias_history, 'Train', 'blue'),
            (self.val_bias_history, 'Validation', 'orange'),
            (self.test_bias_history, 'Test', 'green')
        ]:
            if history:
                epochs = [h['epoch'] for h in history]
                attr_bias = [h.get('average_attribute_bias', 0) for h in history]
                axes[1, 0].plot(epochs, attr_bias, 'o-', label=label, color=color, linewidth=2, markersize=4)
        
        axes[1, 0].set_title('Average Attribute Bias Evolution')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Avg Attribute Bias')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy vs Bias trade-off
        for history, label, color in [
            (self.train_bias_history, 'Train', 'blue'),
            (self.val_bias_history, 'Validation', 'orange'),
            (self.test_bias_history, 'Test', 'green')
        ]:
            if history:
                accuracies = [h.get('overall_accuracy', 0) for h in history]
                biases = [h.get('race_gender_overall_bias', 0) for h in history]
                axes[1, 1].scatter(biases, accuracies, label=label, color=color, alpha=0.7, s=50)
        
        axes[1, 1].set_title('Accuracy vs Bias Trade-off')
        axes[1, 1].set_xlabel('Race-Gender Overall Bias')
        axes[1, 1].set_ylabel('Overall Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / f"bias_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Bias evolution plot saved to: {save_path}")
    
    def plot_subgroup_accuracy_heatmap(self, save_path=None, split='val', latest_epoch_only=True):
        """Plot heatmap of subgroup accuracies."""
        history = getattr(self, f'{split}_bias_history', [])
        if not history:
            print(f"No {split} bias history to plot")
            return
        
        # Get data for heatmap
        if latest_epoch_only:
            data_points = [history[-1]]  # Only latest epoch
            title_suffix = f"(Latest Epoch: {history[-1]['epoch']})"
        else:
            data_points = history  # All epochs
            title_suffix = "(All Epochs)"
        
        # Extract subgroup accuracies
        all_subgroups = set()
        for data in data_points:
            subgroup_accs = data.get('race_gender_subgroup_accuracies', {})
            all_subgroups.update(subgroup_accs.keys())
        
        if not all_subgroups:
            print(f"No subgroup accuracy data found for {split}")
            return
        
        subgroups = sorted(list(all_subgroups))
        
        if latest_epoch_only:
            # Single epoch heatmap
            accuracies = []
            for subgroup in subgroups:
                acc = data_points[0].get('race_gender_subgroup_accuracies', {}).get(subgroup, None)
                accuracies.append(acc if acc is not None else 0.0)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create heatmap data
            heatmap_data = np.array(accuracies).reshape(-1, 1)
            
            # Create heatmap
            sns.heatmap(heatmap_data.T, 
                       annot=True, 
                       fmt='.3f',
                       xticklabels=[sg.replace('Ground Truth ', '').replace('_', '\n') for sg in subgroups],
                       yticklabels=[f'Epoch {data_points[0]["epoch"]}'],
                       cmap='RdYlBu_r',
                       vmin=0, vmax=1,
                       ax=ax,
                       cbar_kws={'label': 'Accuracy'})
            
            ax.set_title(f'{split.title()} Race-Gender Subgroup Accuracies {title_suffix}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Race-Gender Subgroups')
            
        else:
            # Multi-epoch heatmap
            epochs = [data['epoch'] for data in data_points]
            heatmap_data = []
            
            for data in data_points:
                epoch_accuracies = []
                for subgroup in subgroups:
                    acc = data.get('race_gender_subgroup_accuracies', {}).get(subgroup, None)
                    epoch_accuracies.append(acc if acc is not None else 0.0)
                heatmap_data.append(epoch_accuracies)
            
            fig, ax = plt.subplots(figsize=(12, max(8, len(epochs) * 0.5)))
            
            # Create heatmap
            sns.heatmap(np.array(heatmap_data), 
                       annot=True, 
                       fmt='.3f',
                       xticklabels=[sg.replace('Ground Truth ', '').replace('_', '\n') for sg in subgroups],
                       yticklabels=[f'Epoch {e}' for e in epochs],
                       cmap='RdYlBu_r',
                       vmin=0, vmax=1,
                       ax=ax,
                       cbar_kws={'label': 'Accuracy'})
            
            ax.set_title(f'{split.title()} Race-Gender Subgroup Accuracies Evolution', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Race-Gender Subgroups')
            ax.set_ylabel('Training Epochs')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path is None:
            epoch_str = "latest" if latest_epoch_only else "evolution"
            save_path = self.save_dir / f"subgroup_heatmap_{split}_{epoch_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Subgroup accuracy heatmap saved to: {save_path}")
    
    def plot_attribute_bias_comparison(self, save_path=None):
        """Plot per-attribute bias comparison across splits."""
        # Get latest epoch data for each split
        latest_data = {}
        for split, history in [
            ('train', self.train_bias_history),
            ('val', self.val_bias_history), 
            ('test', self.test_bias_history)
        ]:
            if history:
                latest_data[split] = history[-1]
        
        if not latest_data:
            print("No bias data to plot")
            return
        
        # Extract attribute bias data
        all_attributes = set()
        for data in latest_data.values():
            per_attr_bias = data.get('per_attribute_bias', {})
            all_attributes.update(per_attr_bias.keys())
        
        if not all_attributes:
            print("No per-attribute bias data found")
            return
        
        attributes = sorted(list(all_attributes))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Per-attribute bias comparison
        x_pos = np.arange(len(attributes))
        width = 0.25
        
        for i, (split, data) in enumerate(latest_data.items()):
            biases = []
            for attr in attributes:
                bias = data.get('per_attribute_bias', {}).get(attr, 0)
                biases.append(bias)
            
            ax1.bar(x_pos + i * width, biases, width, label=split.title(), alpha=0.8)
        
        ax1.set_title('Per-Attribute Bias Comparison (Latest Epoch)', fontweight='bold')
        ax1.set_xlabel('Attributes')
        ax1.set_ylabel('Bias (Max Accuracy Difference)')
        ax1.set_xticks(x_pos + width)
        ax1.set_xticklabels([attr.replace('Ground Truth ', '') for attr in attributes], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Overall bias summary
        bias_types = ['Race-Gender Overall', 'Race-Gender Avg Subgroup', 'Average Attribute']
        
        for i, (split, data) in enumerate(latest_data.items()):
            bias_values = [
                data.get('race_gender_overall_bias', 0),
                data.get('race_gender_average_subgroup_bias', 0),
                data.get('average_attribute_bias', 0)
            ]
            
            x_pos_summary = np.arange(len(bias_types))
            ax2.bar(x_pos_summary + i * width, bias_values, width, label=split.title(), alpha=0.8)
        
        ax2.set_title('Bias Summary Comparison (Latest Epoch)', fontweight='bold')
        ax2.set_xlabel('Bias Type')
        ax2.set_ylabel('Bias Score')
        ax2.set_xticks(x_pos_summary + width)
        ax2.set_xticklabels(bias_types, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / f"attribute_bias_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Attribute bias comparison plot saved to: {save_path}")
    
    def generate_bias_summary_report(self):
        """Generate a comprehensive bias summary report."""
        print("\n" + "="*80)
        print("🎯 BIAS METRICS TRAINING SUMMARY REPORT")
        print("="*80)
        
        # Training overview
        total_epochs = len(self.overall_metrics_history)
        print(f"Total epochs tracked: {total_epochs}")
        print(f"Splits tracked: Train={len(self.train_bias_history)}, Val={len(self.val_bias_history)}, Test={len(self.test_bias_history)}")
        
        if not self.overall_metrics_history:
            print("No bias data available for summary")
            return
        
        # Latest epoch summary
        latest = self.overall_metrics_history[-1]
        print(f"\n📊 Latest Epoch ({latest['epoch']}) Summary:")
        
        for split in ['train', 'val', 'test']:
            acc = latest.get(f'{split}_accuracy')
            bias = latest.get(f'{split}_bias')
            if acc is not None and bias is not None:
                print(f"  {split.title()}: Accuracy={acc:.2f}%, Overall Bias={bias:.4f}")
        
        # Bias trends
        if total_epochs >= 2:
            print(f"\n📈 Bias Trends (First → Latest Epoch):")
            
            for split in ['train', 'val', 'test']:
                history = getattr(self, f'{split}_bias_history', [])
                if len(history) >= 2:
                    first_bias = history[0].get('race_gender_overall_bias', 0)
                    last_bias = history[-1].get('race_gender_overall_bias', 0)
                    change = last_bias - first_bias
                    direction = "📈 Increased" if change > 0 else "📉 Decreased" if change < 0 else "➡️ Stable"
                    print(f"  {split.title()}: {first_bias:.4f} → {last_bias:.4f} ({change:+.4f}) {direction}")
        
        # Best performance summary
        if self.val_bias_history:
            # Find epoch with best bias-accuracy trade-off (lowest bias with high accuracy)
            best_tradeoff = None
            best_score = float('inf')
            
            for data in self.val_bias_history:
                accuracy = data.get('overall_accuracy', 0)
                bias = data.get('race_gender_overall_bias', 1)
                # Simple trade-off score: bias penalty weighted by accuracy
                score = bias / max(accuracy, 0.01)  # Avoid division by zero
                
                if score < best_score:
                    best_score = score
                    best_tradeoff = data
            
            if best_tradeoff:
                print(f"\n🏆 Best Bias-Accuracy Trade-off (Validation):")
                print(f"  Epoch: {best_tradeoff['epoch']}")
                print(f"  Accuracy: {best_tradeoff.get('overall_accuracy', 0)*100:.2f}%")
                print(f"  Overall Bias: {best_tradeoff.get('race_gender_overall_bias', 0):.4f}")
                print(f"  Avg Subgroup Bias: {best_tradeoff.get('race_gender_average_subgroup_bias', 0):.4f}")
        
        print("="*80)
    
    def save_bias_data(self, filename=None):
        """Save all bias tracking data to JSON file."""
        if filename is None:
            filename = self.save_dir / f"bias_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare data for JSON serialization
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
        
        data = {
            'train_bias_history': make_json_serializable(self.train_bias_history),
            'val_bias_history': make_json_serializable(self.val_bias_history),
            'test_bias_history': make_json_serializable(self.test_bias_history),
            'overall_metrics_history': make_json_serializable(self.overall_metrics_history),
            'metadata': {
                'total_epochs': len(self.overall_metrics_history),
                'generation_time': datetime.now().isoformat(),
                'description': 'Bias metrics tracking data from hierarchical deepfake training'
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Bias tracking data saved to: {filename}")
        
        except Exception as e:
            print(f"❌ Error saving bias data: {e}")
            # Try pickle as fallback
            try:
                import pickle
                pickle_filename = str(filename).replace('.json', '.pkl')
                with open(pickle_filename, 'wb') as f:
                    pickle.dump(data, f)
                print(f"💾 Bias tracking data saved as pickle to: {pickle_filename}")
            except Exception as pickle_error:
                print(f"❌ Failed to save bias data in any format: {pickle_error}")
    
    def generate_all_plots(self):
        """Generate all bias visualization plots."""
        print(f"\n🎨 Generating comprehensive bias visualization plots...")
        
        try:
            # Main bias evolution plot
            self.plot_bias_evolution()
            
            # Subgroup heatmaps for each split
            for split in ['train', 'val', 'test']:
                if getattr(self, f'{split}_bias_history', []):
                    self.plot_subgroup_accuracy_heatmap(split=split, latest_epoch_only=True)
                    if len(getattr(self, f'{split}_bias_history', [])) > 1:
                        self.plot_subgroup_accuracy_heatmap(split=split, latest_epoch_only=False)
            
            # Attribute bias comparison
            self.plot_attribute_bias_comparison()
            
            # Save data
            self.save_bias_data()
            
            # Generate summary report
            self.generate_bias_summary_report()
            
            print(f"✅ All bias visualization plots generated successfully!")
            
        except Exception as e:
            print(f"❌ Error generating bias plots: {e}")
            import traceback
            traceback.print_exc() 