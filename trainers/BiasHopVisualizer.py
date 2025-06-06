import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from pathlib import Path

class BiasHopVisualizer:
    """
    Specialized visualizer for bias hop data from IValueTraversalClusterHop.
    Tracks how I-values change across different subgroups during bias hops.
    """
    
    def __init__(self, save_dir="bias_hop_visualizations"):
        """Initialize the bias hop visualizer."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Store bias metrics history for correlation analysis
        self.validation_bias_history = []
        
    def log_validation_bias_metrics(self, epoch, bias_metrics, subgroup_i_values=None):
        """
        Log validation bias metrics for correlation analysis.
        
        Args:
            epoch: Validation epoch number (should be non-negative integer)
            bias_metrics: Bias metrics from evaluate_model()
            subgroup_i_values: Dictionary of subgroup -> average I-value
        """
        if bias_metrics and epoch >= 0:
            validation_data = {
                'epoch': int(epoch),  # Ensure integer epoch
                'race_gender_overall_bias': bias_metrics.get('race_gender_overall_bias', 0),
                'race_gender_average_subgroup_bias': bias_metrics.get('race_gender_average_subgroup_bias', 0),
                'average_attribute_bias': bias_metrics.get('average_attribute_bias', 0),
                'subgroup_accuracies': bias_metrics.get('race_gender_subgroup_accuracies', {}),
                'subgroup_i_values': subgroup_i_values or {}
            }
            self.validation_bias_history.append(validation_data)
    
    def plot_i_value_statistics_per_hop(self, hop_history, save_path=None):
        """
        Plot I-value statistics (mean, std, range) per hop - keeping this as requested.
        
        Args:
            hop_history: List of dictionaries from traversal.get_hop_i_value_history()
            save_path: Optional path to save the plot
        """
        if not hop_history:
            print("No bias hop history to visualize")
            return
            
        hop_numbers = []
        hop_means = []
        hop_stds = []
        hop_ranges = []
        
        for hop_idx, hop_stats in enumerate(hop_history):
            if hop_stats:
                values = list(hop_stats.values())
                if values:
                    hop_numbers.append(hop_idx)
                    hop_means.append(np.mean(values))
                    hop_stds.append(np.std(values))
                    hop_ranges.append(max(values) - min(values))
        
        if not hop_numbers:
            print("No valid hop data for I-value statistics")
            return
            
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot mean with std dev bands
        ax.plot(hop_numbers, hop_means, 'b-', linewidth=2, marker='o', label='Mean I-value')
        ax.fill_between(hop_numbers,
                       [m - s for m, s in zip(hop_means, hop_stds)],
                       [m + s for m, s in zip(hop_means, hop_stds)],
                       alpha=0.3, color='blue', label='± Std Dev')
        
        # Create secondary y-axis for range
        ax2 = ax.twinx()
        ax2.bar(hop_numbers, hop_ranges, alpha=0.4, color='red', width=0.6, label='I-value Range')
        
        ax.set_title('I-value Statistics per Bias Hop', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hop Number')
        ax.set_ylabel('Mean I-value', color='blue')
        ax2.set_ylabel('I-value Range (Max - Min)', color='red')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / f"i_value_stats_per_hop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ I-value statistics per hop plot saved to: {save_path}")
        
    def plot_subgroup_bias_per_validation_epoch(self, save_path=None):
        """
        Plot subgroup bias metrics evolution per validation epoch.
        """
        if not self.validation_bias_history:
            print("No validation bias history to plot")
            return
            
        epochs = [data['epoch'] for data in self.validation_bias_history]
        overall_bias = [data['race_gender_overall_bias'] for data in self.validation_bias_history]
        avg_subgroup_bias = [data['race_gender_average_subgroup_bias'] for data in self.validation_bias_history]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Overall bias evolution
        axes[0].plot(epochs, overall_bias, 'r-', linewidth=2, marker='o', markersize=4)
        axes[0].set_title('Race-Gender Overall Bias per Validation Epoch', fontweight='bold')
        axes[0].set_xlabel('Validation Epoch')
        axes[0].set_ylabel('Overall Bias (Max Acc Diff)')
        axes[0].grid(True, alpha=0.3)
        
        # Ensure x-axis shows only integer epochs
        if epochs:
            axes[0].set_xticks(range(min(epochs), max(epochs) + 1))
        
        # Average subgroup bias evolution
        axes[1].plot(epochs, avg_subgroup_bias, 'b-', linewidth=2, marker='s', markersize=4)
        axes[1].set_title('Race-Gender Average Subgroup Bias per Validation Epoch', fontweight='bold')
        axes[1].set_xlabel('Validation Epoch')
        axes[1].set_ylabel('Avg Subgroup Bias')
        axes[1].grid(True, alpha=0.3)
        
        # Ensure x-axis shows only integer epochs
        if epochs:
            axes[1].set_xticks(range(min(epochs), max(epochs) + 1))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / f"subgroup_bias_per_epoch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Subgroup bias per validation epoch plot saved to: {save_path}")
    
    def plot_i_value_bias_correlation(self, save_path=None):
        """
        Plot how average I-value of nodes in subgroups correlates with subgroup bias metrics.
        Replaces the unclear "bias reduction" plots.
        """
        if not self.validation_bias_history:
            print("No validation bias history for correlation analysis")
            return
            
        # Collect correlation data
        correlation_data = []
        
        for epoch_data in self.validation_bias_history:
            subgroup_accs = epoch_data.get('subgroup_accuracies', {})
            subgroup_i_vals = epoch_data.get('subgroup_i_values', {})
            epoch = epoch_data['epoch']
            
            # Calculate bias for each subgroup (difference from overall accuracy)
            if subgroup_accs:
                overall_acc = np.mean([acc for acc in subgroup_accs.values() if acc is not None])
                
                for subgroup, accuracy in subgroup_accs.items():
                    if accuracy is not None and subgroup in subgroup_i_vals:
                        i_value = subgroup_i_vals[subgroup]
                        bias = abs(accuracy - overall_acc)  # Individual subgroup bias
                        
                        correlation_data.append({
                            'epoch': epoch,
                            'subgroup': subgroup,
                            'i_value': i_value,
                            'subgroup_bias': bias,
                            'accuracy': accuracy
                        })
        
        if not correlation_data:
            print("No correlation data available")
            return
            
        df = pd.DataFrame(correlation_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot: I-value vs Subgroup Bias
        scatter = axes[0].scatter(df['i_value'], df['subgroup_bias'], 
                                 c=df['epoch'], cmap='viridis', alpha=0.7, s=60)
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['i_value'], df['subgroup_bias'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df['i_value'].min(), df['i_value'].max(), 100)
            axes[0].plot(x_trend, p(x_trend), "r--", alpha=0.8, 
                        label=f'Trend (slope: {z[0]:.4f})')
            axes[0].legend()
        
        axes[0].set_title('Subgroup I-value vs Bias Correlation', fontweight='bold')
        axes[0].set_xlabel('Average Subgroup I-value')
        axes[0].set_ylabel('Subgroup Bias (|Acc - Overall Acc|)')
        axes[0].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label('Validation Epoch')
        
        # Box plot: I-value distribution by bias level
        # Create bias level categories
        df['bias_level'] = pd.cut(df['subgroup_bias'], bins=3, labels=['Low Bias', 'Medium Bias', 'High Bias'])
        
        if not df['bias_level'].isna().all():
            df.boxplot(column='i_value', by='bias_level', ax=axes[1])
            axes[1].set_title('I-value Distribution by Bias Level', fontweight='bold')
            axes[1].set_xlabel('Bias Level')
            axes[1].set_ylabel('Average I-value')
            axes[1].grid(True, alpha=0.3)
            # Remove the automatic title from pandas boxplot
            plt.suptitle('')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / f"i_value_bias_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ I-value bias correlation plot saved to: {save_path}")
    
    def plot_subgroup_targeting_analysis(self, hop_history, save_path=None):
        """
        Analyze which subgroups are being targeted most frequently during hops.
        Fixed x-axis text length issues.
        
        Args:
            hop_history: List of dictionaries from traversal.get_hop_i_value_history()
            save_path: Optional path to save the plot
        """
        if not hop_history:
            print("No bias hop history for targeting analysis")
            return
            
        # Count how often each subgroup has the highest I-value
        subgroup_max_counts = defaultdict(int)
        subgroup_appearances = defaultdict(int)
        
        for hop_stats in hop_history:
            if hop_stats:
                max_i_value = max(hop_stats.values())
                for subgroup, i_value in hop_stats.items():
                    subgroup_appearances[subgroup] += 1
                    if i_value == max_i_value:
                        subgroup_max_counts[subgroup] += 1
        
        # Calculate targeting ratios
        targeting_ratios = {}
        for subgroup in subgroup_appearances:
            targeting_ratios[subgroup] = (subgroup_max_counts[subgroup] / 
                                        subgroup_appearances[subgroup] * 100)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Frequency of being highest I-value
        subgroups = list(subgroup_max_counts.keys())
        max_counts = list(subgroup_max_counts.values())
        
        # Fix x-axis text length issues
        short_labels = []
        for sg in subgroups:
            # Simplify subgroup names
            if 'Ground Truth' in sg:
                sg = sg.replace('Ground Truth ', '').replace('_', '-')
            short_labels.append(sg[:8])  # Limit to 8 characters
        
        axes[0].bar(range(len(subgroups)), max_counts, alpha=0.7)
        axes[0].set_title('Frequency of Highest I-value by Subgroup', fontweight='bold')
        axes[0].set_xlabel('Subgroup')
        axes[0].set_ylabel('Number of Times Highest')
        axes[0].set_xticks(range(len(subgroups)))
        axes[0].set_xticklabels(short_labels, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Targeting ratio (percentage of appearances where subgroup had max I-value)
        ratios = [targeting_ratios.get(sg, 0) for sg in subgroups]
        
        axes[1].bar(range(len(subgroups)), ratios, alpha=0.7, color='orange')
        axes[1].set_title('Targeting Ratio by Subgroup', fontweight='bold')
        axes[1].set_xlabel('Subgroup')
        axes[1].set_ylabel('% of Appearances with Max I-value')
        axes[1].set_xticks(range(len(subgroups)))
        axes[1].set_xticklabels(short_labels, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / f"subgroup_targeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Subgroup targeting analysis plot saved to: {save_path}")
        
    def generate_hop_summary_report(self, hop_history):
        """Generate a text summary of bias hop behavior."""
        if not hop_history:
            print("No bias hop history for summary")
            return
            
        print("\n" + "="*50)
        print("BIAS HOP SUMMARY REPORT")
        print("="*50)
        
        print(f"Total bias hops recorded: {len(hop_history)}")
        
        # Calculate overall statistics
        all_subgroups = set()
        all_i_values = []
        for hop_stats in hop_history:
            all_subgroups.update(hop_stats.keys())
            all_i_values.extend(hop_stats.values())
        
        if all_i_values:
            print(f"Unique subgroups observed: {len(all_subgroups)}")
            print(f"Overall I-value range: {min(all_i_values):.4f} - {max(all_i_values):.4f}")
            print(f"Mean I-value across all hops: {np.mean(all_i_values):.4f}")
            
            # I-value variance trends
            i_value_ranges = []
            for hop_stats in hop_history:
                if hop_stats:
                    values = list(hop_stats.values())
                    i_value_ranges.append(max(values) - min(values))
            
            if len(i_value_ranges) > 1:
                initial_range = i_value_ranges[0]
                final_range = i_value_ranges[-1]
                range_change = final_range - initial_range
                
                print(f"\nI-value Range Evolution:")
                print(f"  Initial range: {initial_range:.4f}")
                print(f"  Final range: {final_range:.4f}")
                print(f"  Change: {range_change:+.4f}")
                print(f"  Trend: {'Increasing' if range_change > 0 else 'Decreasing' if range_change < 0 else 'Stable'}")
        
        # Validation bias summary
        if self.validation_bias_history:
            print(f"\nValidation Bias Tracking:")
            print(f"  Epochs tracked: {len(self.validation_bias_history)}")
            if len(self.validation_bias_history) > 1:
                first_bias = self.validation_bias_history[0]['race_gender_overall_bias']
                last_bias = self.validation_bias_history[-1]['race_gender_overall_bias']
                bias_change = last_bias - first_bias
                print(f"  Overall bias change: {bias_change:+.4f}")
        
        print("="*50) 