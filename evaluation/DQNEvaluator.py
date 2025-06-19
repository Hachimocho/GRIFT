import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.calibration import calibration_curve
import pandas as pd
import json
import os
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional
import pickle

class DQNEvaluator:
    """Comprehensive evaluation framework for DQN performance metrics."""
    
    def __init__(self, save_dir="evaluation_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Validation tracking
        self.validation_history = {
            'loss': [],
            'q_value_mse': [],
            'reward_correlation': [],
            'calibration_error': [],
            'timestamps': []
        }
        
        # Training stability tracking
        self.training_stability = {
            'loss_variance': deque(maxlen=100),
            'gradient_norms': deque(maxlen=100),
            'learning_rate': deque(maxlen=100)
        }
        
    def evaluate_dqn_performance(self, dqn_model, validation_data, epoch=None) -> Dict:
        """Comprehensive DQN performance evaluation."""
        
        metrics = {}
        
        # Basic validation loss
        val_loss = self._calculate_validation_loss(dqn_model, validation_data)
        metrics['val_loss'] = val_loss
        
        # Q-value prediction accuracy
        q_value_mse = self._calculate_q_value_accuracy(dqn_model, validation_data)
        metrics['q_value_mse'] = q_value_mse
        
        # Reward correlation analysis
        reward_correlation = self._calculate_reward_correlation(dqn_model, validation_data)
        metrics['reward_correlation'] = reward_correlation
        
        # Prediction calibration
        calibration_error = self._calculate_calibration_error(dqn_model, validation_data)
        metrics['prediction_calibration'] = calibration_error
        
        # Training stability analysis
        stability_metrics = self._analyze_training_stability()
        metrics.update(stability_metrics)
        
        # Store in history
        self.validation_history['loss'].append(val_loss)
        self.validation_history['q_value_mse'].append(q_value_mse)
        self.validation_history['reward_correlation'].append(reward_correlation)
        self.validation_history['calibration_error'].append(calibration_error)
        self.validation_history['timestamps'].append(datetime.now())
        
        # Save metrics
        if epoch is not None:
            self._save_metrics(metrics, epoch)
            
        return metrics
    
    def _calculate_validation_loss(self, dqn_model, validation_data) -> float:
        """Calculate validation loss on held-out data."""
        dqn_model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch_features, batch_embeddings, batch_rewards in validation_data:
                if batch_embeddings is not None:
                    q_values = dqn_model(batch_features, batch_embeddings)
                else:
                    q_values = dqn_model(batch_features)
                
                loss = torch.nn.functional.mse_loss(q_values, batch_rewards)
                total_loss += loss.item()
                count += 1
        
        dqn_model.train()
        return total_loss / max(count, 1)
    
    def _calculate_q_value_accuracy(self, dqn_model, validation_data) -> float:
        """Calculate how well Q-values predict actual rewards."""
        dqn_model.eval()
        predicted_q = []
        actual_rewards = []
        
        with torch.no_grad():
            for batch_features, batch_embeddings, batch_rewards in validation_data:
                if batch_embeddings is not None:
                    q_values = dqn_model(batch_features, batch_embeddings)
                else:
                    q_values = dqn_model(batch_features)
                
                predicted_q.extend(q_values.cpu().numpy().flatten())
                actual_rewards.extend(batch_rewards.cpu().numpy().flatten())
        
        dqn_model.train()
        
        if len(predicted_q) > 0:
            return mean_squared_error(actual_rewards, predicted_q)
        return float('inf')
    
    def _calculate_reward_correlation(self, dqn_model, validation_data) -> float:
        """Calculate correlation between predicted Q-values and actual rewards."""
        dqn_model.eval()
        predicted_q = []
        actual_rewards = []
        
        with torch.no_grad():
            for batch_features, batch_embeddings, batch_rewards in validation_data:
                if batch_embeddings is not None:
                    q_values = dqn_model(batch_features, batch_embeddings)
                else:
                    q_values = dqn_model(batch_features)
                
                predicted_q.extend(q_values.cpu().numpy().flatten())
                actual_rewards.extend(batch_rewards.cpu().numpy().flatten())
        
        dqn_model.train()
        
        if len(predicted_q) > 1:
            correlation, _ = pearsonr(predicted_q, actual_rewards)
            return correlation if not np.isnan(correlation) else 0.0
        return 0.0
    
    def _calculate_calibration_error(self, dqn_model, validation_data) -> float:
        """Calculate prediction calibration error."""
        dqn_model.eval()
        predicted_probs = []
        binary_outcomes = []
        
        with torch.no_grad():
            for batch_features, batch_embeddings, batch_rewards in validation_data:
                # Get I-value predictions
                i_values = dqn_model.predict_i_value(batch_features, batch_embeddings)
                predicted_probs.extend(i_values.cpu().numpy().flatten())
                
                # Convert rewards to binary outcomes (positive reward = informative)
                binary_outcomes.extend((batch_rewards.cpu().numpy().flatten() > 0).astype(int))
        
        dqn_model.train()
        
        if len(predicted_probs) > 10:  # Need enough samples for calibration
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    binary_outcomes, predicted_probs, n_bins=10
                )
                # Expected calibration error
                return np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            except:
                return float('inf')
        return float('inf')
    
    def _analyze_training_stability(self) -> Dict:
        """Analyze training stability metrics."""
        stability_metrics = {}
        
        # Loss variance
        if len(self.training_stability['loss_variance']) > 10:
            recent_losses = list(self.training_stability['loss_variance'])
            stability_metrics['loss_variance'] = np.var(recent_losses[-20:])
            stability_metrics['loss_trend'] = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        else:
            stability_metrics['loss_variance'] = 0.0
            stability_metrics['loss_trend'] = 0.0
        
        # Gradient norm statistics
        if len(self.training_stability['gradient_norms']) > 5:
            grad_norms = list(self.training_stability['gradient_norms'])
            stability_metrics['avg_gradient_norm'] = np.mean(grad_norms[-10:])
            stability_metrics['gradient_norm_variance'] = np.var(grad_norms[-10:])
        else:
            stability_metrics['avg_gradient_norm'] = 0.0
            stability_metrics['gradient_norm_variance'] = 0.0
        
        return stability_metrics
    
    def update_training_stats(self, loss, model, learning_rate):
        """Update training statistics for stability analysis."""
        self.training_stability['loss_variance'].append(loss)
        self.training_stability['learning_rate'].append(learning_rate)
        
        # Calculate gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.training_stability['gradient_norms'].append(total_norm)
    
    def plot_validation_history(self, save_path=None):
        """Plot validation metrics over time."""
        if len(self.validation_history['loss']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Validation loss
        axes[0, 0].plot(self.validation_history['loss'])
        axes[0, 0].set_title('Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Q-value MSE
        axes[0, 1].plot(self.validation_history['q_value_mse'])
        axes[0, 1].set_title('Q-Value Prediction MSE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].grid(True)
        
        # Reward correlation
        axes[1, 0].plot(self.validation_history['reward_correlation'])
        axes[1, 0].set_title('Reward Correlation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].grid(True)
        
        # Calibration error
        axes[1, 1].plot(self.validation_history['calibration_error'])
        axes[1, 1].set_title('Calibration Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'validation_history.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_metrics(self, metrics, epoch):
        """Save metrics to JSON file."""
        metrics_with_epoch = {'epoch': epoch, **metrics, 'timestamp': datetime.now().isoformat()}
        
        metrics_file = os.path.join(self.save_dir, 'dqn_metrics.jsonl')
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics_with_epoch) + '\n')


class IValueQualityAnalyzer:
    """Comprehensive I-value prediction quality assessment."""
    
    def __init__(self, save_dir="ivalue_analysis"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Track I-value predictions and outcomes
        self.prediction_history = {
            'predicted_i_values': [],
            'actual_outcomes': [],
            'node_attributes': [],
            'timestamps': []
        }
        
        # Traversal efficiency tracking
        self.traversal_metrics = {
            'steps_to_high_value': [],
            'coverage_efficiency': [],
            'attribute_diversity': []
        }
    
    def analyze_i_value_quality(self, nodes, predicted_i_values, actual_outcomes, 
                               epoch=None) -> Dict:
        """Comprehensive I-value quality analysis."""
        
        metrics = {}
        
        # Basic correlation analysis
        correlation_metrics = self._calculate_correlation_metrics(predicted_i_values, actual_outcomes)
        metrics.update(correlation_metrics)
        
        # Precision at k analysis
        precision_metrics = self._calculate_precision_at_k(predicted_i_values, actual_outcomes)
        metrics.update(precision_metrics)
        
        # Traversal efficiency analysis
        efficiency_metrics = self._measure_traversal_efficiency(nodes, predicted_i_values)
        metrics.update(efficiency_metrics)
        
        # Attribute-specific performance
        attribute_metrics = self._analyze_attribute_specific_performance(nodes, predicted_i_values, actual_outcomes)
        metrics.update(attribute_metrics)
        
        # Temporal consistency analysis
        consistency_metrics = self._analyze_temporal_consistency(predicted_i_values)
        metrics.update(consistency_metrics)
        
        # Update history
        self.prediction_history['predicted_i_values'].extend(predicted_i_values)
        self.prediction_history['actual_outcomes'].extend(actual_outcomes)
        self.prediction_history['node_attributes'].extend([node.attributes for node in nodes])
        self.prediction_history['timestamps'].extend([datetime.now()] * len(nodes))
        
        # Save metrics
        if epoch is not None:
            self._save_ivalue_metrics(metrics, epoch)
            
        return metrics
    
    def _calculate_correlation_metrics(self, predicted_i_values, actual_outcomes) -> Dict:
        """Calculate various correlation metrics."""
        metrics = {}
        
        if len(predicted_i_values) > 1:
            # Pearson correlation
            pearson_corr, pearson_p = pearsonr(predicted_i_values, actual_outcomes)
            metrics['pearson_correlation'] = pearson_corr if not np.isnan(pearson_corr) else 0.0
            metrics['pearson_p_value'] = pearson_p if not np.isnan(pearson_p) else 1.0
            
            # Spearman correlation (rank-based)
            spearman_corr, spearman_p = spearmanr(predicted_i_values, actual_outcomes)
            metrics['spearman_correlation'] = spearman_corr if not np.isnan(spearman_corr) else 0.0
            metrics['spearman_p_value'] = spearman_p if not np.isnan(spearman_p) else 1.0
            
            # R-squared
            r2 = r2_score(actual_outcomes, predicted_i_values)
            metrics['r2_score'] = r2 if not np.isnan(r2) else 0.0
            
            # Mean absolute error
            mae = mean_absolute_error(actual_outcomes, predicted_i_values)
            metrics['mean_absolute_error'] = mae
            
        else:
            metrics.update({
                'pearson_correlation': 0.0, 'pearson_p_value': 1.0,
                'spearman_correlation': 0.0, 'spearman_p_value': 1.0,
                'r2_score': 0.0, 'mean_absolute_error': float('inf')
            })
        
        return metrics
    
    def _calculate_precision_at_k(self, predicted_i_values, actual_outcomes, 
                                 k_values=[5, 10, 20]) -> Dict:
        """Calculate precision at k for different k values."""
        metrics = {}
        
        if len(predicted_i_values) == 0:
            for k in k_values:
                metrics[f'precision_at_{k}'] = 0.0
            return metrics
        
        # Sort by predicted I-values (descending)
        sorted_indices = np.argsort(predicted_i_values)[::-1]
        sorted_outcomes = np.array(actual_outcomes)[sorted_indices]
        
        for k in k_values:
            if k <= len(sorted_outcomes):
                # Top k predictions
                top_k_outcomes = sorted_outcomes[:k]
                # Precision = fraction of top k that are actually informative
                precision = np.mean(top_k_outcomes > 0.5)  # Assuming >0.5 is "informative"
                metrics[f'precision_at_{k}'] = precision
            else:
                metrics[f'precision_at_{k}'] = 0.0
        
        return metrics
    
    def _measure_traversal_efficiency(self, nodes, predicted_i_values) -> Dict:
        """Measure how efficiently I-values guide traversal."""
        metrics = {}
        
        if len(nodes) == 0:
            return {'traversal_efficiency': 0.0, 'high_value_coverage': 0.0}
        
        # Simulate traversal efficiency
        # Sort nodes by predicted I-value
        sorted_indices = np.argsort(predicted_i_values)[::-1]
        
        # Calculate how quickly we find high-value nodes
        actual_values = [1.0 if hasattr(node, 'get_label') and node.get_label() == 1 else 0.0 
                        for node in nodes]
        
        high_value_nodes = np.sum(actual_values)
        if high_value_nodes > 0:
            # Steps needed to find 50% of high-value nodes
            cumulative_found = 0
            steps_to_half = len(nodes)
            
            for i, idx in enumerate(sorted_indices):
                if actual_values[idx] > 0.5:
                    cumulative_found += 1
                if cumulative_found >= high_value_nodes * 0.5:
                    steps_to_half = i + 1
                    break
            
            efficiency = 1.0 - (steps_to_half / len(nodes))
            metrics['traversal_efficiency'] = efficiency
            
            # Coverage of high-value nodes in top 20%
            top_20_percent = max(1, len(nodes) // 5)
            top_indices = sorted_indices[:top_20_percent]
            high_value_in_top = sum(1 for idx in top_indices if actual_values[idx] > 0.5)
            coverage = high_value_in_top / high_value_nodes
            metrics['high_value_coverage'] = coverage
            
        else:
            metrics['traversal_efficiency'] = 0.0
            metrics['high_value_coverage'] = 0.0
        
        return metrics
    
    def _analyze_attribute_specific_performance(self, nodes, predicted_i_values, 
                                              actual_outcomes) -> Dict:
        """Analyze I-value prediction performance by node attributes."""
        metrics = {}
        
        # Group by common attributes
        attribute_groups = defaultdict(lambda: {'predicted': [], 'actual': []})
        
        for node, pred, actual in zip(nodes, predicted_i_values, actual_outcomes):
            if hasattr(node, 'attributes'):
                # Group by gender
                if 'gender' in node.attributes:
                    gender = node.attributes['gender']
                    attribute_groups[f'gender_{gender}']['predicted'].append(pred)
                    attribute_groups[f'gender_{gender}']['actual'].append(actual)
                
                # Group by race
                if 'race' in node.attributes:
                    race = node.attributes['race']
                    attribute_groups[f'race_{race}']['predicted'].append(pred)
                    attribute_groups[f'race_{race}']['actual'].append(actual)
                
                # Group by label
                if hasattr(node, 'get_label'):
                    label = node.get_label()
                    attribute_groups[f'label_{label}']['predicted'].append(pred)
                    attribute_groups[f'label_{label}']['actual'].append(actual)
        
        # Calculate correlations for each group
        for group_name, group_data in attribute_groups.items():
            if len(group_data['predicted']) > 5:  # Need enough samples
                corr, _ = pearsonr(group_data['predicted'], group_data['actual'])
                metrics[f'{group_name}_correlation'] = corr if not np.isnan(corr) else 0.0
                metrics[f'{group_name}_count'] = len(group_data['predicted'])
        
        return metrics
    
    def _analyze_temporal_consistency(self, predicted_i_values) -> Dict:
        """Analyze temporal consistency of I-value predictions."""
        metrics = {}
        
        # For now, analyze consistency within current batch
        if len(predicted_i_values) > 1:
            # Coefficient of variation (std/mean)
            cv = np.std(predicted_i_values) / (np.mean(predicted_i_values) + 1e-8)
            metrics['prediction_coefficient_variation'] = cv
            
            # Prediction entropy (measure of uncertainty)
            # Normalize predictions to probabilities
            probs = np.array(predicted_i_values)
            probs = probs / (np.sum(probs) + 1e-8)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            metrics['prediction_entropy'] = entropy
        else:
            metrics['prediction_coefficient_variation'] = 0.0
            metrics['prediction_entropy'] = 0.0
        
        return metrics
    
    def plot_i_value_analysis(self, save_path=None):
        """Plot comprehensive I-value analysis."""
        if len(self.prediction_history['predicted_i_values']) < 10:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        predicted = self.prediction_history['predicted_i_values']
        actual = self.prediction_history['actual_outcomes']
        
        # Scatter plot: predicted vs actual
        axes[0, 0].scatter(predicted, actual, alpha=0.6)
        axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Predicted I-Value')
        axes[0, 0].set_ylabel('Actual Outcome')
        axes[0, 0].set_title('Predicted vs Actual')
        axes[0, 0].grid(True)
        
        # Distribution of predicted I-values
        axes[0, 1].hist(predicted, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Predicted I-Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Predicted I-Values')
        axes[0, 1].grid(True)
        
        # Distribution of actual outcomes
        axes[0, 2].hist(actual, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Actual Outcome')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Distribution of Actual Outcomes')
        axes[0, 2].grid(True)
        
        # Precision at different thresholds
        thresholds = np.linspace(0, 1, 11)
        precisions = []
        for threshold in thresholds:
            high_pred_indices = np.array(predicted) >= threshold
            if np.sum(high_pred_indices) > 0:
                precision = np.mean(np.array(actual)[high_pred_indices] > 0.5)
            else:
                precision = 0.0
            precisions.append(precision)
        
        axes[1, 0].plot(thresholds, precisions, 'b-o')
        axes[1, 0].set_xlabel('I-Value Threshold')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision at Different Thresholds')
        axes[1, 0].grid(True)
        
        # ROC-like curve for I-value prediction
        sorted_indices = np.argsort(predicted)[::-1]
        sorted_actual = np.array(actual)[sorted_indices]
        tpr = np.cumsum(sorted_actual > 0.5) / np.sum(sorted_actual > 0.5)
        fpr = np.cumsum(sorted_actual <= 0.5) / np.sum(sorted_actual <= 0.5)
        
        axes[1, 1].plot(fpr, tpr, 'b-')
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.8)
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC-like Curve for I-Value Prediction')
        axes[1, 1].grid(True)
        
        # Correlation over time (if enough data)
        if len(predicted) > 50:
            window_size = 25
            correlations = []
            windows = []
            for i in range(window_size, len(predicted), 10):
                window_pred = predicted[i-window_size:i]
                window_actual = actual[i-window_size:i]
                corr, _ = pearsonr(window_pred, window_actual)
                correlations.append(corr if not np.isnan(corr) else 0.0)
                windows.append(i)
            
            axes[1, 2].plot(windows, correlations, 'g-o')
            axes[1, 2].set_xlabel('Sample Number')
            axes[1, 2].set_ylabel('Rolling Correlation')
            axes[1, 2].set_title(f'Rolling Correlation (window={window_size})')
            axes[1, 2].grid(True)
        else:
            axes[1, 2].text(0.5, 0.5, 'Insufficient data\nfor rolling correlation', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Rolling Correlation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'ivalue_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_ivalue_metrics(self, metrics, epoch):
        """Save I-value metrics to file."""
        metrics_with_epoch = {'epoch': epoch, **metrics, 'timestamp': datetime.now().isoformat()}
        
        metrics_file = os.path.join(self.save_dir, 'ivalue_metrics.jsonl')
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics_with_epoch) + '\n')


class DQNComparisonFramework:
    """Framework for comparing different DQN architectures."""
    
    def __init__(self, save_dir="dqn_comparison"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.comparison_results = {}
        self.model_profiles = {}
    
    def compare_dqn_models(self, models_dict, test_data, num_trials=3) -> Dict:
        """Comprehensive comparison of different DQN models."""
        
        results = {}
        
        for model_name, model in models_dict.items():
            print(f"Evaluating model: {model_name}")
            
            model_results = []
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}")
                
                # Accuracy metrics
                accuracy_metrics = self._evaluate_accuracy(model, test_data)
                
                # Efficiency metrics
                efficiency_metrics = self._evaluate_efficiency(model, test_data)
                
                # Robustness metrics
                robustness_metrics = self._evaluate_robustness(model, test_data)
                
                # Computational cost
                cost_metrics = self._profile_computational_cost(model, test_data)
                
                trial_results = {
                    **accuracy_metrics,
                    **efficiency_metrics,
                    **robustness_metrics,
                    **cost_metrics
                }
                model_results.append(trial_results)
            
            # Aggregate results across trials
            results[model_name] = self._aggregate_trial_results(model_results)
        
        # Statistical comparison
        statistical_results = self._statistical_comparison(results)
        
        # Save results
        self._save_comparison_results(results, statistical_results)
        
        return {'model_results': results, 'statistical_comparison': statistical_results}
    
    def _evaluate_accuracy(self, model, test_data) -> Dict:
        """Evaluate model accuracy metrics."""
        model.eval()
        
        predicted_i_values = []
        actual_outcomes = []
        q_values = []
        rewards = []
        
        with torch.no_grad():
            for batch_features, batch_embeddings, batch_rewards in test_data:
                # Get I-value predictions
                i_vals = model.predict_i_value(batch_features, batch_embeddings)
                predicted_i_values.extend(i_vals.cpu().numpy().flatten())
                
                # Get Q-values
                if batch_embeddings is not None:
                    q_vals = model(batch_features, batch_embeddings)
                else:
                    q_vals = model(batch_features)
                q_values.extend(q_vals.cpu().numpy().flatten())
                
                # Store rewards and convert to binary outcomes
                batch_rewards_np = batch_rewards.cpu().numpy().flatten()
                rewards.extend(batch_rewards_np)
                actual_outcomes.extend((batch_rewards_np > 0).astype(int))
        
        # Calculate accuracy metrics
        metrics = {}
        
        if len(predicted_i_values) > 0:
            # I-value correlation with outcomes
            i_corr, _ = pearsonr(predicted_i_values, actual_outcomes)
            metrics['i_value_correlation'] = i_corr if not np.isnan(i_corr) else 0.0
            
            # Q-value correlation with rewards
            q_corr, _ = pearsonr(q_values, rewards)
            metrics['q_value_correlation'] = q_corr if not np.isnan(q_corr) else 0.0
            
            # MSE for both
            metrics['i_value_mse'] = mean_squared_error(actual_outcomes, predicted_i_values)
            metrics['q_value_mse'] = mean_squared_error(rewards, q_values)
            
            # Precision at k
            sorted_indices = np.argsort(predicted_i_values)[::-1]
            for k in [5, 10, 20]:
                if k <= len(sorted_indices):
                    top_k_outcomes = np.array(actual_outcomes)[sorted_indices[:k]]
                    metrics[f'precision_at_{k}'] = np.mean(top_k_outcomes)
                else:
                    metrics[f'precision_at_{k}'] = 0.0
        
        model.train()
        return metrics
    
    def _evaluate_efficiency(self, model, test_data) -> Dict:
        """Evaluate computational efficiency."""
        model.eval()
        
        # Measure inference time
        inference_times = []
        memory_usage = []
        
        for batch_features, batch_embeddings, _ in test_data:
            # Measure inference time
            start_time = time.time()
            
            with torch.no_grad():
                _ = model.predict_i_value(batch_features, batch_embeddings)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Measure memory usage (if CUDA available)
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated())
        
        model.train()
        
        metrics = {
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'throughput_samples_per_sec': len(test_data[0][0]) / np.mean(inference_times) if inference_times else 0
        }
        
        if memory_usage:
            metrics['avg_memory_usage_mb'] = np.mean(memory_usage) / (1024 * 1024)
        
        return metrics
    
    def _evaluate_robustness(self, model, test_data) -> Dict:
        """Evaluate model robustness to noise and perturbations."""
        model.eval()
        
        # Test with noise
        noise_levels = [0.01, 0.05, 0.1]
        robustness_metrics = {}
        
        # Get baseline predictions
        baseline_predictions = []
        with torch.no_grad():
            for batch_features, batch_embeddings, _ in test_data:
                preds = model.predict_i_value(batch_features, batch_embeddings)
                baseline_predictions.extend(preds.cpu().numpy().flatten())
        
        # Test robustness to feature noise
        for noise_level in noise_levels:
            noisy_predictions = []
            
            with torch.no_grad():
                for batch_features, batch_embeddings, _ in test_data:
                    # Add noise to features
                    noise = torch.randn_like(batch_features) * noise_level
                    noisy_features = batch_features + noise
                    
                    preds = model.predict_i_value(noisy_features, batch_embeddings)
                    noisy_predictions.extend(preds.cpu().numpy().flatten())
            
            # Calculate correlation between baseline and noisy predictions
            if len(baseline_predictions) > 0:
                corr, _ = pearsonr(baseline_predictions, noisy_predictions)
                robustness_metrics[f'robustness_noise_{noise_level}'] = corr if not np.isnan(corr) else 0.0
        
        model.train()
        return robustness_metrics
    
    def _profile_computational_cost(self, model, test_data) -> Dict:
        """Profile computational cost of the model."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory footprint
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        # Measure forward pass FLOPs (simplified estimation)
        # This is a rough estimation based on linear layer sizes
        estimated_flops = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                estimated_flops += module.in_features * module.out_features
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'estimated_flops': estimated_flops
        }
    
    def _aggregate_trial_results(self, trial_results) -> Dict:
        """Aggregate results across multiple trials."""
        if not trial_results:
            return {}
        
        aggregated = {}
        
        # Get all metric names
        all_metrics = set()
        for trial in trial_results:
            all_metrics.update(trial.keys())
        
        # Calculate mean and std for each metric
        for metric in all_metrics:
            values = [trial.get(metric, 0.0) for trial in trial_results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_values'] = values
        
        return aggregated
    
    def _statistical_comparison(self, results) -> Dict:
        """Perform statistical comparison between models."""
        from scipy.stats import ttest_ind
        
        statistical_results = {}
        model_names = list(results.keys())
        
        # Compare each pair of models
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f'{model1}_vs_{model2}'
                statistical_results[comparison_key] = {}
                
                # Compare key metrics
                key_metrics = ['i_value_correlation_mean', 'q_value_correlation_mean', 
                              'precision_at_10_mean', 'avg_inference_time_mean']
                
                for metric in key_metrics:
                    if metric in results[model1] and metric in results[model2]:
                        values1 = results[model1][metric.replace('_mean', '_values')]
                        values2 = results[model2][metric.replace('_mean', '_values')]
                        
                        if len(values1) > 1 and len(values2) > 1:
                            t_stat, p_value = ttest_ind(values1, values2)
                            statistical_results[comparison_key][metric] = {
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
        
        return statistical_results
    
    def _save_comparison_results(self, results, statistical_results):
        """Save comparison results to files."""
        # Save detailed results
        with open(os.path.join(self.save_dir, 'model_comparison_results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for model_name, model_results in results.items():
                serializable_results[model_name] = {}
                for metric, value in model_results.items():
                    if isinstance(value, np.ndarray):
                        serializable_results[model_name][metric] = value.tolist()
                    elif isinstance(value, list):
                        serializable_results[model_name][metric] = value
                    else:
                        serializable_results[model_name][metric] = float(value) if np.isscalar(value) else value
            
            json.dump(serializable_results, f, indent=2)
        
        # Save statistical comparison
        with open(os.path.join(self.save_dir, 'statistical_comparison.json'), 'w') as f:
            json.dump(statistical_results, f, indent=2)
        
        # Create summary CSV
        self._create_summary_csv(results)
    
    def _create_summary_csv(self, results):
        """Create a summary CSV of model comparison."""
        summary_data = []
        
        for model_name, model_results in results.items():
            row = {'Model': model_name}
            
            # Key metrics to include in summary
            key_metrics = [
                'i_value_correlation_mean', 'q_value_correlation_mean',
                'precision_at_10_mean', 'avg_inference_time_mean',
                'total_parameters', 'model_size_mb'
            ]
            
            for metric in key_metrics:
                if metric in model_results:
                    row[metric] = model_results[metric]
                else:
                    row[metric] = None
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(self.save_dir, 'model_comparison_summary.csv'), index=False)
    
    def plot_comparison_results(self, results, save_path=None):
        """Plot comparison results."""
        if len(results) < 2:
            return
        
        model_names = list(results.keys())
        key_metrics = ['i_value_correlation_mean', 'q_value_correlation_mean', 
                      'precision_at_10_mean', 'avg_inference_time_mean']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(key_metrics):
            if i >= len(axes):
                break
                
            means = []
            stds = []
            names = []
            
            for model_name in model_names:
                if metric in results[model_name]:
                    means.append(results[model_name][metric])
                    std_metric = metric.replace('_mean', '_std')
                    stds.append(results[model_name].get(std_metric, 0))
                    names.append(model_name)
            
            if means:
                x_pos = np.arange(len(names))
                axes[i].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
                axes[i].set_xlabel('Model')
                axes[i].set_ylabel(metric.replace('_', ' ').title())
                axes[i].set_title(f'Comparison: {metric.replace("_", " ").title()}')
                axes[i].set_xticks(x_pos)
                axes[i].set_xticklabels(names, rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'model_comparison_plot.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close() 