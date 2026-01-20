# HyperGraph Documentation

Welcome to the HyperGraph documentation. This documentation is organized to mirror the codebase structure for easy navigation.

## Overview

HyperGraph is a research framework for graph-based training of deepfake detection models with bias-aware capabilities. The framework uses I-value estimation via Deep Q-Networks (DQN) to identify and correct demographic bias in real-time.

## Documentation Structure

### Core Components

- **[Managers](managers/README.md)** - Graph management classes including reduction and restoration strategies
- **[Web UI](web_ui/README.md)** - Web-based configuration and results interface
- **[Features](features/)** - Feature documentation and usage guides

### Key Features

- **Graph Reduction & Restoration** - Dynamic graph modification strategies during training
- **I-Value Based Training** - DQN-guided exploration and bias correction
- **Adaptive Traversal** - Dynamic switching between traversal methods
- **Model Rollback** - Automatic checkpoint restoration on validation drops

## Quick Links

- [Graph Reduction & Restoration Guide](features/graph_reduction_restoration.md)
- [GraphReductionManager API](managers/GraphReductionManager.md)
- [Web UI Configuration](web_ui/configuration.md)
- [Web UI Reduction/Restoration](web_ui/graph_reduction_restoration.md)

## Getting Started

For new users, start with:
1. [Graph Reduction & Restoration Guide](features/graph_reduction_restoration.md) - Overview of reduction/restoration features
2. [Web UI Configuration](web_ui/configuration.md) - How to configure experiments via the web interface
