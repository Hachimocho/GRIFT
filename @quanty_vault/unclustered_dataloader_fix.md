# UnclusteredDeepfakeDataloader Fix - Quanty 8

## Problem
The `UnclusteredDeepfakeDataloader` was missing the `_build_graph_standard` method that was being called in the cache generation code in `web_ui/app.py`, causing an AttributeError.

## Solution
Updated the `UnclusteredDeepfakeDataloader` to match the structure and methods of the `HierarchicalDeepfakeDataloader`:

### Key Changes Made:

1. **Removed LSH-specific code**: Removed `use_lsh`, `lsh_bands`, `lsh_band_size` hyperparameters and related methods
2. **Updated hyperparameters**: Made them consistent with hierarchical implementation
3. **Added missing methods**:
   - `_extract_attribute_matrices()` - for vectorized similarity calculations
   - `_calculate_similarity()` - for individual node comparisons
   - `_calculate_pairwise_similarities()` - for vectorized batch processing
   - `_filter_edges_vectorized()` - for efficient edge filtering
   - `_filter_edges()` - main filtering method
   - `_apply_attribute_filtering()` - applies all filters in sequence
   - `_create_graph_from_edges()` - creates graph from filtered edges
   - `_build_graph_standard()` - main graph building method
   - `_build_graph()` - entry point for graph building
   - `get_graph()` - retrieves graphs for specific splits

4. **Updated `load()` method**: Made it consistent with hierarchical implementation
5. **Removed old methods**: Removed `_create_attribute_matrix()`, `_compute_similarity()`, `_hierarchical_match()`, and `process_node_batch()` which were from the old implementation

### Key Differences from Hierarchical Implementation:

1. **No clustering**: The unclustered version generates all possible edge pairs instead of clustering by race/gender/age
2. **No subgroup mapping**: The `node_index_to_subgroup_id` parameter is ignored in unclustered methods
3. **Simpler fallback**: Disconnected nodes are connected randomly to any other node instead of within subgroups

### Cache Generation Compatibility:
The unclustered dataloader now has the same interface as the hierarchical one, so the cache generation code in `web_ui/app.py` will work correctly with both implementations.

## Testing
The fix should resolve the AttributeError when using `graph_type='nonclustered'` in cache generation. 