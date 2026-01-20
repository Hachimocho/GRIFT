# Graph Reduction & Restoration Feature

## Overview

Graph Reduction & Restoration is a feature that allows dynamic modification of the training graph during training. Nodes can be removed from the graph using various strategies, and restored when validation performance drops.

## Purpose and Motivation

The goal of graph reduction and restoration is to:
- **Focus Training**: Remove nodes that are less informative or redundant
- **Adaptive Learning**: Dynamically adjust the training set based on model performance
- **Bias Correction**: Use I-values to identify and remove biased nodes
- **Performance Recovery**: Restore nodes when model performance degrades

## Reduction Strategies

### Max I-value Reduction

Removes the top X% of nodes by I-value (highest I-values first).

**When to Use:**
- When you want to focus on nodes the model struggles with (high I-value = low expected performance)
- To reduce training on nodes the model already handles well

**Requirements:**
- I-value traversal method must be selected
- DQN must be training/predicting

### Min I-value Reduction

Removes the bottom Y% of nodes by I-value (lowest I-values first).

**When to Use:**
- When you want to focus on challenging nodes
- To remove nodes the model handles very well (low I-value = high expected performance)

**Requirements:**
- I-value traversal method must be selected
- DQN must be training/predicting

### Mix-Max I-value Reduction

Removes both top X% and bottom Y% of nodes by I-value (mutually exclusive).

**When to Use:**
- When you want to remove both very easy and very hard nodes
- To focus training on nodes with moderate difficulty

**Requirements:**
- I-value traversal method must be selected
- DQN must be training/predicting

### Random Reduction

Removes Z% of nodes randomly.

**When to Use:**
- As a baseline/control method
- When I-values are not available
- To test the effect of graph size reduction without I-value bias

**Requirements:**
- None (works with any traversal method)

## Restoration Strategies

### Random Pool Restoration

Restores a random selection of removed nodes from the pool.

**When to Use:**
- As a baseline restoration method
- When I-values are not available for removed nodes
- For simple recovery when performance drops

**Requirements:**
- Removed nodes must exist in the pool

### Targeted Restoration

Restores nodes with I-values closest to the average I-value of all removed nodes.

**When to Use:**
- When you want to restore "average" nodes
- To maintain a balanced distribution of node difficulties
- When I-values are available for removed nodes

**Requirements:**
- I-value traversal method must be selected
- I-values must be stored for removed nodes

### Reversion Restoration

Restores the nodes that were removed in the previous epoch.

**When to Use:**
- When you want to "undo" the previous epoch's reduction
- To test if removing nodes was beneficial
- For conservative restoration that preserves recent history

**Requirements:**
- Epoch removal history must be maintained

## Model Rollback Feature

Separate from graph reduction/restoration, model rollback automatically reloads the best model checkpoint when validation accuracy drops below the best seen so far.

**When to Use:**
- To prevent model degradation
- When you want to ensure training always starts from the best known state
- As a safety mechanism during exploration

**Configuration:**
- Enable Model Rollback: Turn the feature on/off
- Rollback on Validation Drop: Automatically rollback when validation drops

## Configuration Examples

### Example 1: Aggressive High I-value Reduction

```json
{
  "reduction_enabled": true,
  "reduction_strategy": "max_ival",
  "reduction_percentage": 20.0,
  "reduction_interval": "end_of_epoch",
  "restoration_enabled": true,
  "restoration_strategy": "random_pool",
  "restoration_percentage": 50.0,
  "restoration_trigger_threshold": 0.02
}
```

**Use Case**: Remove 20% of highest I-value nodes each epoch, restore 50% randomly if validation drops by 2% or more.

### Example 2: Conservative Mix-Max Reduction

```json
{
  "reduction_enabled": true,
  "reduction_strategy": "mix_max_ival",
  "reduction_top_percentage": 5.0,
  "reduction_bottom_percentage": 5.0,
  "reduction_interval": "end_of_epoch",
  "restoration_enabled": true,
  "restoration_strategy": "targeted",
  "restoration_percentage": 30.0,
  "restoration_trigger_threshold": 0.01
}
```

**Use Case**: Remove 5% top + 5% bottom I-value nodes each epoch, restore 30% of average I-value nodes if validation drops by 1% or more.

### Example 3: Random Baseline with Model Rollback

```json
{
  "reduction_enabled": true,
  "reduction_strategy": "random",
  "reduction_percentage": 10.0,
  "reduction_interval": "end_of_epoch",
  "restoration_enabled": false,
  "model_rollback_enabled": true,
  "model_rollback_on_val_drop": true
}
```

**Use Case**: Remove 10% of nodes randomly each epoch, no restoration, but rollback model if validation drops.

## Best Practices

1. **Start with Random**: Use random reduction first to establish a baseline
2. **Monitor Validation**: Watch validation accuracy closely when using reduction
3. **Tune Thresholds**: Adjust restoration trigger thresholds based on your dataset
4. **Use I-values Carefully**: I-value-based strategies require DQN to be well-trained
5. **Combine with Model Rollback**: Use model rollback as a safety net
6. **Experiment Gradually**: Start with small percentages and increase gradually
7. **Track Statistics**: Monitor reduction/restoration statistics in logs

## When to Use Each Strategy

### Use Max I-value Reduction When:
- Model is overfitting to easy nodes
- You want to focus on challenging examples
- I-values are reliable and well-calibrated

### Use Min I-value Reduction When:
- Model needs more challenging examples
- You want to remove redundant easy nodes
- Training is too slow due to easy nodes

### Use Mix-Max I-value Reduction When:
- You want a balanced approach
- Both easy and hard nodes are problematic
- You want to focus on moderate difficulty nodes

### Use Random Reduction When:
- Testing the effect of graph size reduction
- I-values are not available
- You need a baseline comparison

### Use Random Pool Restoration When:
- Simple recovery is sufficient
- I-values are not available
- You want a baseline restoration method

### Use Targeted Restoration When:
- You want to maintain balanced node distribution
- I-values are available and reliable
- You want intelligent node selection

### Use Reversion Restoration When:
- You want to test if reduction was beneficial
- Conservative restoration is preferred
- You want to preserve recent history

## Troubleshooting

**Reduction not happening:**
- Check that reduction is enabled
- Verify reduction percentage > 0
- Check reduction interval settings
- Ensure graph has enough nodes

**Restoration not triggering:**
- Verify restoration is enabled
- Check validation accuracy is actually dropping
- Lower restoration trigger threshold
- Ensure removed nodes exist in pool

**I-value errors:**
- Ensure I-value traversal is selected
- Check that DQN is training
- Verify trainer has get_i_value method
- Check I-value predictions are valid

**Performance degradation:**
- Reduce reduction percentage
- Lower restoration trigger threshold
- Enable model rollback
- Check reduction strategy appropriateness
