# Graph Reduction & Restoration UI

This document describes the web UI interface for configuring graph reduction and restoration strategies.

## Configuration Section

The "Graph Reduction & Restoration" section appears on the configuration page (`/configure`) and includes three main subsections:

### 1. Graph Reduction Configuration

**Enable Graph Reduction**: Checkbox to enable/disable graph reduction

When enabled, the following options become available:

- **Reduction Strategy**: Select from:
  - None: No reduction
  - Max I-value: Remove top X% nodes by I-value
  - Min I-value: Remove bottom Y% nodes by I-value
  - Mix-Max I-value: Remove top X% + bottom Y% (mutually exclusive)
  - Random: Remove Z% randomly (baseline)

- **Reduction Percentage**: Percentage of nodes to remove (0-100)

- **Mix-Max Configuration** (shown when Mix-Max strategy is selected):
  - Top Percentage: Percentage of top I-value nodes to remove
  - Bottom Percentage: Percentage of bottom I-value nodes to remove

- **Reduction Interval**: When to perform reduction
  - End of Epoch: At the end of each training epoch
  - Every N Steps: Periodically during training

- **Steps Between Reductions** (shown when "Every N Steps" is selected):
  - Number of training steps between reductions

### 2. Node Restoration Configuration

**Enable Node Restoration**: Checkbox to enable/disable node restoration

When enabled, the following options become available:

- **Restoration Strategy**: Select from:
  - None: No restoration
  - Random Pool: Restore random selection from removed nodes pool
  - Targeted: Restore nodes with I-values closest to average
  - Reversion: Restore previous epoch's removed nodes

- **Restoration Percentage**: Percentage of removed nodes to restore (0-100)

- **Restoration Trigger Threshold**: Minimum validation accuracy drop to trigger restoration
  - Default: 0.0 (any drop triggers restoration)
  - Higher values require larger drops before restoration

### 3. Model Rollback Configuration

**Enable Model Rollback**: Checkbox to enable/disable model rollback

When enabled:

- **Rollback on Validation Drop**: Checkbox to rollback model when validation accuracy drops below best seen
  - When enabled, the model checkpoint is automatically reloaded from the best checkpoint
  - This happens independently of graph reduction/restoration

## Validation

The UI includes automatic validation:

- **I-Value Requirement**: If non-random reduction OR restoration strategies are selected, at least one I-value traversal method must be selected in the Traversal Configuration section
- **Warning Display**: A warning message appears if I-value traversal is required but not selected
- **Percentage Validation**: Percentage fields are validated to be within 0-100 range

## Results Display

Reduction and restoration strategies are displayed in the results page:

- **Reduction Strategy Column**: Shows the reduction strategy used (or "None")
- **Restoration Strategy Column**: Shows the restoration strategy used (or "None")
- **Comparison Modal**: Includes reduction/restoration strategy information when comparing runs

## Usage Tips

1. **Start Simple**: Begin with random reduction to establish a baseline
2. **I-Value First**: Ensure I-value traversal is working before using I-value-based reduction
3. **Monitor Performance**: Watch validation accuracy to tune restoration thresholds
4. **Experiment**: Try different combinations of reduction and restoration strategies
5. **Model Rollback**: Use model rollback separately from graph reduction/restoration for different purposes

## Troubleshooting

**Warning: "Non-random reduction or restoration strategies require I-value traversal"**
- Solution: Select an I-value traversal method in the Traversal Configuration section

**Reduction not happening**
- Check that reduction is enabled
- Verify reduction percentage is > 0
- Check reduction interval settings

**Restoration not triggering**
- Verify restoration is enabled
- Check that validation accuracy is actually dropping
- Lower restoration trigger threshold if needed
- Ensure there are removed nodes in the pool
