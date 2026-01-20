# Configuration System

The HyperGraph Web UI provides a comprehensive configuration interface for setting up experiments.

## Configuration Page

Access the configuration page at `/configure` or `/configure/<config_name>` to edit an existing configuration.

## Configuration Sections

### Basic Settings

- **Configuration Name**: Unique identifier for the configuration
- **Description**: Human-readable description of the experiment
- **Architecture**: Model architecture(s) to use
- **DQN Model**: DQN model type for I-value prediction

### Traversal Configuration

- **Primary Traversal Type**: Initial traversal method
  - Comprehensive: Systematic node visiting
  - Random: Random walk traversal
  - I-Value: DQN-guided exploration

- **Traversal Switching**: Option to switch between traversal methods during training
  - Second/Third Traversal Types
  - Switch Epochs
  - Disconnected Switching: Reset main model after switch

### Graph Reduction & Restoration

See [Graph Reduction & Restoration UI](graph_reduction_restoration.md) for detailed documentation.

### Cache & Fairness Settings

- **Cache Options**: Use cached nodes, cache nodes, cache full dataset
- **Fair Training/Testing**: Enable balanced sampling by demographic groups

### Visualization Settings

- **Track Nodes Count**: Number of nodes to track for visualization
- **Visualization Sample Size**: Sample size for generating visualizations

## Validation

The configuration system includes validation to ensure:
- Required fields are filled
- I-value traversal is selected when using non-random reduction/restoration
- Cache compatibility with selected settings
- Valid percentage ranges for reduction/restoration

## Saving Configurations

Configurations are saved via the "Save & Run" button, which:
1. Validates the configuration
2. Saves to the configuration store
3. Optionally starts a test run immediately

## Loading Configurations

To edit an existing configuration:
1. Navigate to `/configure/<config_name>`
2. Modify settings as needed
3. Save the updated configuration
