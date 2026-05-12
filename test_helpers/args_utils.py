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
    parser.add_argument('--dynamic-cache-detection', action='store_true',
                        help='Automatically detect cache size from existing cache files (use with --use-cached)')
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

    # Traversal configuration options
    parser.add_argument('--traversal-type', type=str, default='comprehensive',
                        choices=['comprehensive', 'random', 'i-value', 'i-value-cluster-hop', 'i-value-subcluster', 'i-value-cluster-hop-subcluster'],
                        help='Single traversal type to use throughout training (default: comprehensive)')
    
    # Switch traversal mode options  
    parser.add_argument('--enable-traversal-switching', action='store_true',
                        help='Enable dynamic traversal switching during training')
    parser.add_argument('--traversal-sequence', type=str, default='comprehensive,i-value-cluster-hop',
                        help='Comma-separated sequence of traversals for switching mode (default: comprehensive,i-value-cluster-hop)')
    parser.add_argument('--switch-epochs', type=str, default='10',
                        help='Comma-separated epochs at which to switch traversals (default: 10)')
    
    # Architecture testing options
    parser.add_argument('--architectures', type=str, default='vistransformdf',
                        help='Comma-separated list of CNN architectures to test (default: vistransformdf)')
    parser.add_argument('--test-all-traversals', action='store_true',
                        help='Test all traversal types with each architecture (for comparison)')
    
    # I-value visualization options
    parser.add_argument('--enable-ivalue-viz', action='store_true',
                        help='Enable I-value visualization tracking during training')
    parser.add_argument('--viz-sample-size', type=int, default=1000,
                        help='Number of nodes to sample per epoch for I-value statistics (default: 1000)')
    parser.add_argument('--viz-track-nodes', type=int, default=50,
                        help='Number of specific nodes to track throughout training (default: 50)')
    parser.add_argument('--viz-step-frequency', type=int, default=10,
                        help='Log I-value statistics every N training steps (default: 10)')
    parser.add_argument('--viz-save-dir', type=str, default='ivalue_visualizations',
                        help='Directory to save I-value visualization plots (default: ivalue_visualizations)')
    
    # DQN model selection
    parser.add_argument('--dqn-model', type=str, default='basic',
                      choices=['basic', 'residual', 'attention', 'conv_embedding', 'ensemble'],
                      help='Type of DQN model to use for I-value prediction (default: basic)')

    # Test-time uncertainty method selection
    parser.add_argument('--uncertainty-methods', type=str, default='',
                        help='Comma-separated uncertainty methods to calculate during final test evaluation (choices planned: msp,ddu,trust_score,graph)')

    # Run ID for output organization
    parser.add_argument('--run-id', type=str, default=None,
                        help='Unique run ID for organizing outputs (set by web UI)')

    # New argument for disconnected traversal switching
    parser.add_argument('--disconnected-switching', action='store_true',
                        help='If set, resets the main detection model after traversal switching (I-value model is NOT reset). Only relevant if traversal switching is enabled.')

    # Graph construction type
    parser.add_argument('--graph-type', type=str, default='clustered',
                        choices=['clustered', 'clustered_subclustered', 'nonclustered', 'nonclustered_subclustered'],
                        help='Type of graph construction: clustered (race-gender groups), nonclustered (all nodes), and/or with subclustering (Louvain) (default: clustered)')

    parser.add_argument('--export-csv-per-run', dest='export_csv_per_run', action='store_true',
                        help='Export node/edge CSVs with subcluster info for each run (default: True)')
    parser.add_argument('--no-export-csv-per-run', dest='export_csv_per_run', action='store_false',
                        help='Do not export node/edge CSVs for each run')
    parser.set_defaults(export_csv_per_run=True)

    # GPU override options (off by default)
    parser.add_argument('--gpu-override', action='store_true',
                        help='Enable single-GPU override (forces use of a single GPU). Off by default.')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use when --gpu-override is enabled (default: 0)')

    # Traversal steps configuration
    parser.add_argument('--train-steps', type=int, default=1000,
                        help='Number of traversal steps during training (default: 1000)')
    parser.add_argument('--val-steps', type=int, default=1000,
                        help='Number of traversal steps during validation (default: 1000)')
    parser.add_argument('--train-steps-equal-nodes', action='store_true',
                        help='If set, training steps will equal the number of nodes in the train graph')
    parser.add_argument('--val-steps-equal-nodes', action='store_true',
                        help='If set, validation steps will equal the number of nodes in the validation graph')

    # Bias inference controls (default disabled for performance on large graphs)
    parser.add_argument('--enable-train-bias-inference', action='store_true',
                        help='Enable training bias inference (disabled by default)')
    parser.add_argument('--enable-val-bias-inference', action='store_true',
                        help='Enable validation bias inference (disabled by default)')
    
    # Performance optimization options
    parser.add_argument('--val-num-workers', type=int, default=4,
                        help='Number of parallel workers for validation image loading (default: 4)')

    return parser.parse_args()
