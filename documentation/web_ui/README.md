# Web UI Documentation

The HyperGraph Web UI provides a browser-based interface for configuring experiments, monitoring training runs, and analyzing results.

## Components

- **[Configuration System](configuration.md)** - How to configure experiments via the web interface
- **[Graph Reduction & Restoration UI](graph_reduction_restoration.md)** - UI-specific documentation for reduction/restoration features

## Features

- **Configuration Management** - Create, edit, and save experiment configurations
- **Run Management** - Start, monitor, and stop training runs
- **Results Analysis** - Compare results across multiple runs
- **Cache Management** - Generate and manage node/graph caches

## Access

The web UI is typically accessed at `http://localhost:5000` when running locally, or via SSH tunnel for remote access.

## Quick Start

1. Navigate to `/configure` to create a new configuration
2. Fill in the required fields (architecture, traversal, etc.)
3. Configure graph reduction/restoration if desired
4. Save and start a test run
5. Monitor progress at `/runs`
6. View results at `/results`
