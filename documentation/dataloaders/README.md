# Dataloader Classes

Dataloaders load datasets and construct graph structures from raw data. They handle data preprocessing, node creation, and edge construction.

## Available Dataloaders

### HierarchicalDeepfakeDataloader
Hierarchical graph construction with demographic grouping and quality filtering.

**Key Features:**
- Groups nodes by race-gender combinations
- Creates fully-connected subgraphs within groups
- Applies quality/similarity thresholds for cross-group edges
- Supports sparse mode for large graphs

**Documentation:** [HierarchicalDeepfakeDataloader.md](HierarchicalDeepfakeDataloader.md)

### ClusteredDeepfakeDataloader
Creates clustered graphs with demographic-based clustering.

### ConnectedClusteredDeepfakeDataloader
Clustered graphs with guaranteed connectivity.

### UnclusteredDeepfakeDataloader
Simple graph construction without clustering.

## Base Dataloader Class

All dataloaders inherit from `Dataloader` base class, which defines:

```python
class Dataloader:
    def load(self) -> HyperGraph
```

## Graph Construction Process

1. **Load Datasets**: Load raw data from datasets
2. **Create Nodes**: Convert data points to Node objects
3. **Extract Attributes**: Extract demographic and quality attributes
4. **Group Nodes**: Group by demographic attributes
5. **Construct Edges**: Create edges based on similarity/thresholds
6. **Return Graph**: Return HyperGraph instance

## Hierarchical Construction

The hierarchical approach:

1. **Level 1**: Group by categorical attributes (race-gender)
2. **Level 2**: Create fully-connected subgraphs within groups
3. **Level 3**: Apply quality thresholds for cross-group edges

## Edge Construction Strategies

### Intra-Demographic Edges
- Fully connected within demographic groups
- No threshold filtering

### Inter-Demographic Edges
- Quality metric similarity (blur, brightness, contrast)
- Symmetry similarity
- Face embedding similarity (cosine similarity)

### Thresholds
- `embedding_threshold`: Face embedding similarity (default: 0.9)
- `quality_threshold`: Quality metric similarity (default: 0.9)
- `symmetry_threshold`: Symmetry similarity (default: 0.9)

## Usage Example

```python
from dataloaders.HierarchicalDeepfakeDataloader import HierarchicalDeepfakeDataloader
from datasets.AIFaceDataset import AIFaceDataset
from edges.Edge import Edge

# Load datasets
datasets = [AIFaceDataset("path/to/data")]

# Create dataloader
dataloader = HierarchicalDeepfakeDataloader(
    datasets=datasets,
    edge_class=Edge,
    embedding_threshold=0.9,
    quality_threshold=0.9,
    sparse_mode=True
)

# Load graph
graph = dataloader.load()
```

## Performance Optimizations

### Sparse Mode
For large graphs (>5000 nodes per subgroup):
- Uses k-nearest neighbors instead of full connectivity
- Batch processing for memory efficiency
- Configurable k and batch size

### Caching
- Node data is cached for faster access
- Graph structures can be cached to disk
- Cache compatibility checking

### Parallel Processing
- Multi-threaded attribute extraction
- Vectorized similarity calculations
- Chunked processing for memory efficiency

## Configuration Options

### Hyperparameters
- `embedding_threshold`: Face embedding similarity threshold
- `quality_threshold`: Quality metric similarity threshold
- `symmetry_threshold`: Symmetry similarity threshold
- `sparse_mode`: Enable sparse edge generation
- `sparse_k_neighbors`: Number of neighbors for sparse mode
- `sparse_subgroup_threshold`: Subgroup size threshold for sparse mode
- `assign_subclusters`: Enable Louvain subclustering

### Silent Mode
- `silent_mode`: Disable progress bars and verbose logging
- Useful for batch processing and automation

## Notes

- Dataloaders handle data preprocessing and normalization
- Attribute extraction is dataset-specific
- Graph construction can be time-consuming for large datasets
- Caching significantly speeds up repeated runs
