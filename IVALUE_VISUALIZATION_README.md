# I-Value Visualization System

This document describes the I-value visualization system for tracking and analyzing how I-values change during model training with I-value traversal methods.

## Overview

The I-value visualization system provides several approaches to visualize I-value dynamics during training:

1. **Aggregate Statistical Tracking**: Track mean, median, std dev, and distribution statistics over time
2. **Subgroup-based Analysis**: Monitor I-values across different demographic subgroups
3. **Individual Node Tracking**: Follow specific nodes throughout training
4. **Bias Hop Visualization**: Analyze bias reduction patterns in cluster hop traversal
5. **Distribution Snapshots**: Capture I-value distributions at key training moments

## Key Components

### 1. IValueVisualizationTracker (`trainers/IValueVisualizationTracker.py`)

Main visualization tracking class that:
- Collects I-value statistics during training
- Tracks aggregate metrics per epoch and step
- Monitors subgroup-specific I-value patterns
- Manages individual node tracking
- Generates comprehensive visualization plots

### 2. BiasHopVisualizer (`trainers/BiasHopVisualizer.py`)

Specialized visualizer for bias hop data from `IValueTraversalClusterHop`:
- Tracks subgroup targeting patterns
- Analyzes bias reduction over time
- Visualizes hop frequency and effectiveness
- Generates bias evolution reports

## Usage

### Command Line Options

Enable I-value visualization by adding these arguments to `test_hierarchical.py`:

```bash
python test_hierarchical.py \
  --enable-ivalue-viz \
  --traversal-type i-value \
  --viz-sample-size 1000 \
  --viz-track-nodes 50 \
  --viz-step-frequency 10 \
  --viz-save-dir ivalue_visualizations
```

**Visualization Arguments:**
- `--enable-ivalue-viz`: Enable I-value visualization tracking
- `--viz-sample-size`: Number of nodes to sample per epoch for statistics (default: 1000)
- `--viz-track-nodes`: Number of specific nodes to track throughout training (default: 50)
- `--viz-step-frequency`: Log I-value statistics every N training steps (default: 10)
- `--viz-save-dir`: Directory to save visualization plots (default: ivalue_visualizations)

### Quick Start Examples

```bash
# Basic I-value traversal with visualization
python test_hierarchical.py --enable-ivalue-viz --traversal-type i-value --num-epochs 10

# I-value cluster hop with bias analysis
python test_hierarchical.py --enable-ivalue-viz --traversal-type i-value-cluster-hop --bias_hop_period 50

# Traversal switching with visualization
python test_hierarchical.py --enable-ivalue-viz --enable-traversal-switching \
  --traversal-sequence "comprehensive,i-value-cluster-hop" --switch-epochs "5"

# Compare all traversals
python test_hierarchical.py --enable-ivalue-viz --test-all-traversals
```

### Running the Example Script

Use the provided example script for guided demonstrations:

```bash
python example_ivalue_visualization.py
```

This interactive script provides several pre-configured examples with different traversal configurations.

## Generated Visualizations

### 1. Training Progression Plots

**File:** `training_progression_YYYYMMDD_HHMMSS.png`

Four-panel plot showing:
- **Mean I-value ± Std Dev over time**: Central tendency and variability
- **Distribution evolution**: Quartiles, min/max ranges over epochs
- **High/Low I-value ratios**: Proportion of nodes with extreme I-values
- **Sample sizes**: Number of nodes processed per epoch

### 2. Subgroup Analysis Plots

**File:** `subgroup_analysis_YYYYMMDD_HHMMSS.png`

Four-panel plot showing:
- **Mean I-values by subgroup over time**: Bias patterns across demographics
- **I-value standard deviation by subgroup**: Variability within subgroups
- **Latest epoch subgroup comparison**: Current state comparison
- **Subgroup sample sizes**: Data availability per subgroup

### 3. Individual Node Tracking

**File:** `tracked_nodes_YYYYMMDD_HHMMSS.png`

Two-panel plot showing:
- **Individual trajectories**: I-value evolution for specific tracked nodes
- **Average trajectory**: Mean I-value evolution across tracked nodes

### 4. Bias Hop Visualizations (Cluster Hop Only)

#### Bias Hop Evolution
**File:** `bias_hop_evolution_YYYYMMDD_HHMMSS.png`

Four-panel analysis:
- **Subgroup evolution lines**: I-value changes per subgroup during hops
- **Heatmap**: I-value intensity across subgroups and hops
- **Statistics per hop**: Mean and standard deviation trends
- **Bias measure per hop**: Range (max-min) showing bias levels

#### Subgroup Targeting Analysis
**File:** `subgroup_targeting_YYYYMMDD_HHMMSS.png`

Three-panel analysis:
- **Targeting frequency**: How often each subgroup has highest I-value
- **Targeting ratio**: Percentage of appearances with maximum I-value
- **Efficiency scatter**: Relationship between frequency and targeting success

#### Bias Reduction Analysis
**File:** `bias_reduction_YYYYMMDD_HHMMSS.png`

Two-panel analysis:
- **Bias reduction over time**: Trend line showing bias evolution
- **Bias vs Mean I-value**: Relationship between overall I-values and bias

### 5. Raw Data Export

**File:** `ivalue_data_YYYYMMDD_HHMMSS.json`

Complete dataset including:
- Epoch-level statistics
- Step-level tracking data
- Individual node histories
- Distribution snapshots
- Bias hop history

## Interpretation Guide

### Understanding I-Value Patterns

**High I-values (>0.7)**: Indicate high information content or exploration value
**Low I-values (<0.3)**: Suggest well-learned or low-information nodes
**Stable patterns**: May indicate convergence or lack of learning
**Oscillating patterns**: Could suggest active learning or instability

### Subgroup Analysis

**Equal I-values across subgroups**: Suggests unbiased information utilization
**Large subgroup differences**: May indicate demographic bias in information seeking
**Converging trends**: Could show bias reduction over training
**Diverging trends**: May indicate increasing bias

### Bias Hop Effectiveness

**Decreasing bias measure**: Shows successful bias reduction
**Consistent targeting**: Indicates effective subgroup identification
**Range reduction**: Demonstrates fairness improvement

## Scalability Considerations

### For Large Datasets (>1M nodes):

1. **Reduce sample sizes**:
   ```bash
   --viz-sample-size 500 --viz-track-nodes 20
   ```

2. **Increase logging frequency**:
   ```bash
   --viz-step-frequency 50
   ```

3. **Use targeted visualization**:
   - Focus on specific traversal types
   - Limit epoch counts for testing
   - Use bias hop visualization only when needed

### Memory Management

- Sample-based statistics prevent memory overload
- Cached computations reduce redundant I-value calculations
- Configurable history length limits memory usage

## Integration with Existing Code

### Adding to New Trainer Classes

```python
from trainers.IValueVisualizationTracker import IValueVisualizationTracker

class YourTrainer:
    def __init__(self, ...):
        # Initialize visualization tracker
        self.viz_tracker = IValueVisualizationTracker(save_dir="your_viz_dir")
        
    def train_epoch(self, epoch):
        self.viz_tracker.start_epoch(epoch)
        
        # Your training code...
        
        # Log epoch summary
        self.viz_tracker.log_epoch_summary(self, sample_size=1000)
```

### Custom Visualization Extensions

```python
# Extend the tracker for custom metrics
class CustomIValueTracker(IValueVisualizationTracker):
    def log_custom_metric(self, metric_name, value):
        # Add custom tracking logic
        pass
        
    def plot_custom_analysis(self):
        # Add custom visualization plots
        pass
```

## Performance Tips

1. **Optimize sampling**: Use representative samples rather than full datasets
2. **Batch I-value computation**: Collect I-values efficiently during training
3. **Async visualization**: Generate plots in background threads
4. **Selective tracking**: Enable visualization only for I-value traversals
5. **Progressive detail**: Use different detail levels for different stages

## Troubleshooting

### Common Issues

**1. No visualizations generated:**
- Check that `--enable-ivalue-viz` is set
- Verify I-value traversal is being used
- Ensure trainer has `get_i_value` method

**2. Empty plots:**
- Verify nodes have valid I-values
- Check sample size is not too small
- Ensure attribute metadata is properly configured

**3. Memory errors:**
- Reduce `--viz-sample-size` and `--viz-track-nodes`
- Increase `--viz-step-frequency`
- Use shorter training runs for testing

**4. Missing bias hop data:**
- Verify using `i-value-cluster-hop` traversal
- Check `bias_hop_period` is reasonable
- Ensure sufficient training epochs

### Debug Mode

Add debug information to track issues:

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check visualization data
print(f"Epoch stats: {len(viz_tracker.epoch_stats)}")
print(f"Tracked nodes: {len(viz_tracker.tracked_nodes)}")
print(f"Bias hop history: {len(viz_tracker.bias_hop_history)}")
```

## Best Practices

1. **Start small**: Test with short epochs and small samples first
2. **Monitor resources**: Watch memory and disk usage during long runs
3. **Save incrementally**: Generate plots periodically, not just at the end
4. **Use version control**: Track visualization parameters with results
5. **Document experiments**: Keep notes on parameter choices and findings

## Future Extensions

Potential enhancements to the visualization system:

1. **Interactive plots**: Web-based dashboards with zoom/filter capabilities
2. **Real-time monitoring**: Live visualization during training
3. **Comparative analysis**: Side-by-side comparison of different runs
4. **Statistical testing**: Automated significance testing for bias patterns
5. **Predictive modeling**: Forecast I-value trends based on current patterns

This comprehensive visualization system provides insights into I-value dynamics that were previously impossible to observe at scale, enabling better understanding and optimization of I-value-based traversal methods. 