import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test the hierarchical graph construction approach')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited nodes')
    parser.add_argument('--visualize', action='store_true', help='Generate graph visualizations')
    parser.add_argument('--show', action='store_true', help='Show visualizations (requires --visualize)')
    parser.add_argument('--quality-threshold', type=float, default=0.8, 
                        help='Similarity threshold for quality metrics (default: 0.8)')
    parser.add_argument('--symmetry-threshold', type=float, default=0.75, 
                        help='Similarity threshold for facial symmetry (default: 0.75)')
    parser.add_argument('--embedding-threshold', type=float, default=0.7, 
                        help='Similarity threshold for face embeddings (default: 0.7)')
    
    # Node caching options
    parser.add_argument('--cache-nodes', action='store_true', 
                        help='Save loaded nodes to cache file for faster testing')
    parser.add_argument('--cache-full', action='store_true',
                        help='Cache the entire dataset instead of just a subset (use with --cache-nodes)')
    parser.add_argument('--use-cached', action='store_true', 
                        help='Use previously cached nodes instead of loading from dataset')
    parser.add_argument('--use-full-cache', action='store_true',
                        help='Load the full dataset from cache instead of the subset (use with --use-cached)')
    parser.add_argument('--cached-nodes', type=int, default=1000, 
                        help='Number of nodes to cache per split when not using full cache (default: 1000)')
    parser.add_argument('--cache-file', type=str, default='node_cache/cached_nodes.pkl', 
                        help='Filename for caching/loading nodes (relative to script execution dir)')

    # Grid search options
    parser.add_argument('--search', action='store_true',
                        help='Run grid search over threshold combinations')
    parser.add_argument('--search-split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Split to use for grid search (default: train)')
    parser.add_argument('--quality-steps', type=int, default=5,
                        help='Number of steps for quality threshold grid search (default: 5)')
    parser.add_argument('--symmetry-steps', type=int, default=5,
                        help='Number of steps for symmetry threshold grid search (default: 5)')
    parser.add_argument('--embedding-steps', type=int, default=5,
                        help='Number of steps for embedding threshold grid search (default: 5)')
    parser.add_argument('--search-results', type=str, default='threshold_search_results.csv',
                        help='File to save search results to (default: threshold_search_results.csv)')
    
    # Training options
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for training and evaluation (default: 100)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for DataLoader (default: 4)')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--bias_loss_weight', type=float, default=0.00,
                        help='Weight for bias loss (default: 0.00)')
    parser.add_argument('--bias_hop_period', type=int, default=100,
                        help='Period for bias hop (default: 100)')
    parser.add_argument('--load-last-checkpoint', action='store_true',
                        help='Load the last best checkpoint if validation accuracy decreases.')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs (default: logs)')
    parser.add_argument('--fair-train', action='store_true', help='Use subgroup-balanced training set for graph construction')
    parser.add_argument('--fair-test', action='store_true', help='Use subgroup-balanced validation/test sets for graph construction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility') # Add seed argument

    return parser.parse_args()
