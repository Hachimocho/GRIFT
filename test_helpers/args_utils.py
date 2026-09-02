import argparse

#: Traversal names the CLI accepts today.
TRAVERSAL_TYPES = ('comprehensive', 'random', 'i-value')

#: Retired names, and what they resolve to.
#:
#: `i-value-cluster-hop` is a pure rename: hopping is now selected from the graph, because a
#: clustered graph is built from disjoint race-gender groups that a pointer cannot walk
#: between, so it was never an independent strategy -- the old name and `--graph-type
#: clustered` always had to be set together.
#:
#: The two `*-subcluster` names are a **behavior change**, not a rename. Louvain-community
#: area selection is gone. It never ran as designed: its outlier filter excluded every node
#: whenever the variance was zero, `CapabilityManager` never enabled a DQN for it so its
#: I-values were random draws, and it yielded one node per step against ~17 for the other
#: walks. A run that asked for it gets plain `i-value` and a notice saying so.
DEPRECATED_TRAVERSAL_TYPES = {
    'i-value-cluster-hop': 'i-value',
    'i-value-subcluster': 'i-value',
    'i-value-cluster-hop-subcluster': 'i-value',
}

#: Every spelling that maps onto the single I-value traversal.
IVALUE_TRAVERSAL_ALIASES = frozenset(
    {'i-value'} | set(DEPRECATED_TRAVERSAL_TYPES)
)


def canonical_traversal_type(traversal_type, quiet=False):
    """Resolve a traversal name, announcing a retired one. Returns the canonical name."""
    replacement = DEPRECATED_TRAVERSAL_TYPES.get(traversal_type)
    if replacement is None:
        return traversal_type
    if not quiet:
        print(f"NOTE: traversal type {traversal_type!r} has been retired; using "
              f"{replacement!r}.")
        if 'subcluster' in traversal_type:
            print("      Louvain-community area selection was removed -- it selected on "
                  "random I-values, its outlier filter excluded every node at zero "
                  "variance, and it yielded 1 node per step against ~17. This run is NOT "
                  "equivalent to the old one.")
        else:
            print("      Cluster hopping is now chosen from --graph-type, so behavior is "
                  "unchanged for a clustered graph.")
    return replacement


def parse_args(argv=None):
    """Parse the training CLI.

    ``argv=None`` reads ``sys.argv`` exactly as before. Passing a list makes the whole
    training path callable in-process, which is what the sweep driver's synthetic tier
    and the functional tests need -- otherwise every programmatic caller has to mutate
    ``sys.argv`` around the call.
    """
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
    parser.add_argument('--data-root', type=str, default=None,
                        help='Path to the AI-Face dataset root. If omitted, the pipeline will try environment variables and common server paths.')

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
    parser.add_argument('--checkpoint-metric', type=str, default='auroc',
                        choices=['accuracy', 'balanced_accuracy', 'auroc'],
                        help='Validation metric that decides the best epoch. '
                             'accuracy was the original behavior and is a poor choice on '
                             'an imbalanced split: a model that predicts one class for '
                             'every sample scores the majority-class prior (~87%% here) at '
                             'epoch 1 and can never strictly beat it, so the best epoch '
                             'freezes at 1 and everything afterwards -- later training, '
                             'graph rewiring, node reduction -- is computed and then '
                             'discarded. balanced_accuracy is prevalence-free but pins to '
                             'exactly 0.5 for such a model, so it ties instead of '
                             'improving and freezes too. auroc (default) is threshold-free '
                             'and moves whenever the ranking improves, which is why it is '
                             'the default')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs (default: logs)')
    parser.add_argument('--fair-train', action='store_true', help='Use subgroup-balanced (race x gender) training set for graph construction. Does NOT touch the real/fake class balance -- see --balance-labels for that')
    parser.add_argument('--fair-test', action='store_true', help='Use subgroup-balanced (race x gender) validation/test sets for graph construction. Does NOT touch the real/fake class balance')
    parser.add_argument('--balance-labels', type=str, default='none',
                        choices=['none', 'train', 'all'],
                        help='Equalize the real/fake class counts. The corrected AI-Face '
                             'split is ~87%% fake, and at that prior BCE is minimized '
                             'substantially by raising the output bias, so models emit one '
                             'class for every sample: accuracy equals the prior and '
                             'balanced accuracy pins at 0.5. train balances only the '
                             'training set, leaving val/test on the real distribution; all '
                             'balances every split, which makes a 0.5 threshold directly '
                             'interpretable but stops measuring the deployed distribution. '
                             'Only ~13%% of the corpus is real, so balancing discards most '
                             'fakes (default: none)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility') # Add seed argument

    # Traversal configuration options
    parser.add_argument('--traversal-type', type=str, default='comprehensive',
                        choices=list(TRAVERSAL_TYPES) + list(DEPRECATED_TRAVERSAL_TYPES),
                        help='Single traversal type to use throughout training. i-value '
                             'picks its walk from --graph-type: a clustered graph is built '
                             'from disjoint groups so its pointers hop between clusters, an '
                             'unclustered one is connected so they do not. '
                             'i-value-cluster-hop is a retired spelling of i-value; the two '
                             '*-subcluster names are retired and now resolve to i-value '
                             'with a behavior change (default: comprehensive)')
    
    # Switch traversal mode options  
    parser.add_argument('--enable-traversal-switching', action='store_true',
                        help='Enable dynamic traversal switching during training')
    parser.add_argument('--traversal-sequence', type=str, default='comprehensive,i-value',
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
                      choices=['basic', 'residual', 'attention', 'conv_embedding', 'ensemble',
                               'loss_ewma', 'gain_linear', 'gain_residual', 'gain_ensemble'],
                      help="I-value estimator. The five original names are DEPRECATED: "
                           "measured against realised learning gain they rank at Spearman "
                           "+0.01 where a sample's current loss ranks at +0.33. "
                           "'loss_ewma' is the unlearned control -- it ranks candidates by "
                           "the loss each last incurred, a dictionary lookup. "
                           "'gain_linear' IS that same signal but trainable, "
                           "everything else must beat; 'gain_residual' adds a gated learned "
                           "residual on top of it; 'gain_ensemble' averages several heads. "
                           "Pass --dqn-fixes to run a legacy architecture inside the fixed "
                           "base instead. (default: basic)")
    parser.add_argument('--dqn-fixes', action='store_true', default=False,
                      help="Run the selected legacy --dqn-model as the learned residual "
                           "inside the fixed base: direct path for the model-state features, "
                           "a small recency-weighted buffer, raw output and a rank/huber "
                           "objective. No effect on the gain_* models, which are already "
                           "fixed.")
    parser.add_argument('--dqn-objective', type=str, default='rank',
                      choices=['rank', 'huber'],
                      help="Training objective for the fixed estimators. 'rank' is a "
                           "pairwise logistic loss on which of two samples taught more -- "
                           "it matches how the value is consumed (argmax over candidates) "
                           "and is invariant to the target's heavy tail (skew +4.0, kurtosis "
                           "+45). 'huber' regresses signed-log1p(gain) instead, keeping a "
                           "calibrated magnitude. (default: rank)")
    parser.add_argument('--dqn-buffer-size', type=int, default=512,
                      help="Transitions the fixed estimators retain. The old 10,000 held a "
                           "full epoch, but a reward is only valid for the model that "
                           "produced it -- the same node's gain self-correlates only +0.24 "
                           "across two epochs. (default: 512)")
    parser.add_argument('--dqn-embedding-dim', type=int, default=512,
                      help="Face-embedding width fed to the estimator; 0 removes the "
                           "embedding pathway entirely. It was 67%% of the old input while "
                           "the model-state features were 6.3%%, and it is a static property "
                           "of the image that says nothing about what the model now knows.")

    # Run ID for output organization
    parser.add_argument('--run-id', type=str, default=None,
                        help='Unique run ID for organizing outputs (set by web UI)')

    # New argument for disconnected traversal switching
    parser.add_argument('--disconnected-switching', action='store_true',
                        help='If set, resets the main detection model after traversal switching (I-value model is NOT reset). Only relevant if traversal switching is enabled.')

    # Graph updaters. The manager was hardcoded to NoGraphManager, so
    # PerformanceGraphManager was imported and never constructed; the reduction settings
    # were read from a per-configuration dict that nothing ever populated. Both defaults
    # below reproduce the previous behavior exactly.
    parser.add_argument('--graph-manager', type=str, default='none',
                        choices=['none', 'performance'],
                        help='Graph updater applied to the training graph between epochs. '
                             'none (default) leaves the graph static, matching the '
                             'behavior before this flag existed. performance rewires by '
                             'predicted I-value and therefore requires an I-value '
                             'traversal, which is what supplies the predictor')
    parser.add_argument('--weak-quantile', type=float, default=0.9,
                        help='I-value quantile above which a node counts as weak -- the '
                             'model expects to keep learning from it. A quantile, not an '
                             'absolute value: the previous absolute 0.8/0.2 pair did not '
                             'bracket the DQN output at all, so no node was ever classified '
                             'strong and the updater changed nothing (default: 0.9)')
    parser.add_argument('--strong-quantile', type=float, default=0.1,
                        help='Quantile below which a node counts as strong -- already '
                             'learned (default: 0.1)')
    parser.add_argument('--removal-fraction', type=float, default=0.02,
                        help='Share of the training graph withdrawn per update, capped at '
                             '0.05 and never below half the starting size (default: 0.02)')
    parser.add_argument('--graph-updates-per-epoch', type=int, default=4,
                        help='How many times per epoch the graph updater runs. It used to '
                             'tick once per epoch against a step-counted interval, so only '
                             'the epochs after the best checkpoint could matter '
                             '(default: 4)')
    parser.add_argument('--graph-manager-sample-nodes', type=int, default=0,
                        help='Extra nodes sampled per update to measure I-values, on top of '
                             'the ones training already visits. 0 (default) means no extra '
                             'sampling: every node the traversal touches is recorded anyway '
                             'at O(1) memory each, so coverage grows with training instead '
                             'of costing a separate pass of DQN forward passes. Set it '
                             'positive only to seed the quantiles faster at the start of a '
                             'run')
    parser.add_argument('--graph-remove-target', type=str, default='strong',
                        choices=['strong', 'weak', 'random'],
                        help='Which end to withdraw. strong prunes already-learned nodes '
                             '(curriculum); weak prunes the ones the model keeps failing on '
                             '(noise); random is the control -- same budget and schedule, '
                             'chosen without reference to I-values, which is what makes the '
                             'other two attributable to the I-value signal rather than to '
                             'pruning as such (default: strong)')
    # Graph reduction / restoration. Fully implemented in the epoch loop and previously
    # unreachable: the keys were read off the internal per-configuration dict, which is
    # built from these args and never carried them.
    parser.add_argument('--reduction-enabled', action='store_true',
                        help='Enable graph reduction during training (default: disabled)')
    parser.add_argument('--reduction-strategy', type=str, default='none',
                        choices=['none', 'max_ival', 'min_ival', 'mix_max_ival', 'random'],
                        help='Which nodes to remove. The *_ival strategies read '
                             'trainer.get_i_value and so require an I-value traversal; '
                             'random works with any (default: none)')
    parser.add_argument('--reduction-percentage', type=float, default=0.0,
                        help='Percentage of nodes to remove per reduction, 0-100 (default: 0.0)')
    parser.add_argument('--reduction-top-percentage', type=float, default=0.0,
                        help='Top percentage for the mix_max_ival strategy, 0-100 (default: 0.0)')
    parser.add_argument('--reduction-bottom-percentage', type=float, default=0.0,
                        help='Bottom percentage for the mix_max_ival strategy, 0-100 (default: 0.0)')
    parser.add_argument('--reduction-interval', type=str, default='end_of_epoch',
                        choices=['end_of_epoch', 'every_n_steps'],
                        help='When to reduce (default: end_of_epoch)')
    parser.add_argument('--reduction-interval-steps', type=int, default=100,
                        help='Steps between reductions when --reduction-interval is '
                             'every_n_steps (default: 100)')
    parser.add_argument('--restoration-strategy', type=str, default='none',
                        choices=['none', 'random_pool', 'targeted', 'reversion'],
                        help='How to restore removed nodes when validation accuracy '
                             'drops (default: none)')
    parser.add_argument('--restoration-percentage', type=float, default=50.0,
                        help='Percentage of the removed pool to restore, 0-100 (default: 50.0)')
    parser.add_argument('--restoration-trigger-threshold', type=float, default=0.0,
                        help='Minimum validation-accuracy drop that triggers restoration '
                             '(default: 0.0)')

    # Graph construction type
    parser.add_argument('--graph-type', type=str, default='clustered',
                        choices=['clustered', 'clustered_subclustered', 'nonclustered', 'nonclustered_subclustered'],
                        help='Type of graph construction: clustered (race-gender groups), nonclustered (all nodes), and/or with subclustering (Louvain) (default: clustered)')

    parser.add_argument('--edge-construction', type=str, default='knn',
                        choices=['knn', 'all_pairs'],
                        help='How candidate edges are generated before similarity '
                             'filtering. all_pairs was the original behavior and is O(N^2) '
                             'in memory *before* filtering -- ~76 TiB of RAM at the full '
                             '1.6M-node corpus, and the surviving graph was 97%% dense '
                             '(average degree 1264 on a measured 1304-node split). knn '
                             '(default) keeps each node\'s --knn-neighbors nearest '
                             'neighbours by cosine distance over the face embedding, which '
                             'is O(k*N) and sparse by construction. Similarity filtering is '
                             'unchanged; there are simply no longer N^2 chances to pass it. '
                             'all_pairs is refused above 30,000 nodes')
    parser.add_argument('--knn-neighbors', type=int, default=50,
                        help='Neighbours per node under --edge-construction knn. Realised '
                             'degree can exceed this because the graph is symmetrised '
                             '(default: 50)')
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
    parser.add_argument('--ivalue-reward', choices=('confidence', 'learning_gain'),
                        default='confidence',
                        help="What the DQN's reward -- and so an I-value -- means. "
                             "'confidence': +/-confidence by correctness, so a high Q is "
                             "'already mastered' and predict_i_value's 1-sigmoid(Q) turns it "
                             "into 'the model does poorly here'. 'learning_gain': the "
                             "sample's measured loss reduction across its own update, where "
                             "a high Q already means informative and the inversion is undone.")
    parser.add_argument('--ivalue-state-features', action='store_true', default=False,
                        help="Append model-state features (current probability, loss, "
                             "times-seen, staleness) to the DQN input. Without them the "
                             "I-value is a static function of node attributes and cannot "
                             "notice that the model has since learned a sample.")
    parser.add_argument('--ivalue-candidate-pool', type=int, default=0,
                        help="Extra uniformly drawn nodes considered alongside the current "
                             "node's neighbours when picking the next one. 0 keeps the walk "
                             "purely local, where the argmax ranges over ~8 similarity-linked "
                             "and therefore correlated neighbours.")
    parser.add_argument('--comprehensive-cumulative', action='store_true', default=False,
                        help="Let the comprehensive traversal's visited set persist across "
                             "epochs, making it a real exhaustive curriculum. Without this it "
                             "resamples every epoch and can never run out of data.")
    parser.add_argument('--ivalue-selection', choices=('max', 'band', 'min'), default='max',
                        help="How a candidate is chosen. 'max' takes the most informative, "
                             "the historical premise. 'band' draws from a quantile RANGE "
                             "(--ivalue-band), on the hypothesis that the very hardest "
                             "samples are outliers and mislabels on an 87.55%%-imbalanced "
                             "corpus. 'min' is the deliberate opposite, kept as a control. "
                             "'band' needs --ivalue-candidate-pool to mean anything: a "
                             "quantile over ~8 neighbours is noise.")
    parser.add_argument('--ivalue-band', default='0.4,0.7',
                        help="Quantile range for --ivalue-selection band, as low,high.")
    parser.add_argument('--ivalue-loss-weight', choices=('none', 'linear', 'rank'),
                        default='none',
                        help="Scale each sample's loss by its I-value instead of selecting "
                             "with it, keeping i.i.d. sampling and therefore its batch "
                             "diversity. 'rank' is invariant to the estimator's output scale. "
                             "Applies to the basic training path and only with "
                             "--uncertainty-head none.")
    parser.add_argument('--ivalue-weight-clip', type=float, default=2.0,
                        help="Loss weights are bounded to [1/clip, clip], geometric about 1.")
    parser.add_argument('--ivalue-ban-negative-gain', type=int, default=0,
                        help="Withdraw a node once its mean MEASURED gain has stayed negative "
                             "over this many visits (0 disables). 38%% of trained samples "
                             "currently show a negative gain. Requires --ivalue-reward "
                             "learning_gain, which is what measures it.")
    parser.add_argument('--ivalue-ban-max-fraction', type=float, default=0.2,
                        help="Ceiling on the fraction of the graph banning may withdraw, so "
                             "the arm keeps measuring 'train on data that helps' rather than "
                             "'train on less data'.")
    parser.add_argument('--ivalue-group-targeting', action='store_true', default=False,
                        help="Rank demographic groups by mean I-value and draw candidates only "
                             "from the weakest ones, sampling UNIFORMLY within them. Group "
                             "means cancel the per-sample label noise that instance-level "
                             "selection amplifies, and uniform sampling inside a group keeps "
                             "the batch diversity that selection destroys.")
    parser.add_argument('--ivalue-group-top', type=int, default=3,
                        help="How many groups --ivalue-group-targeting keeps.")
    parser.add_argument('--ivalue-unseen-prior',
                        choices=('neutral', 'optimistic', 'pessimistic'), default='neutral',
                        help="How an UNVISITED node's loss placeholder ranks against visited "
                             "ones. This silently decided the meaning of an experiment: the "
                             "neutral placeholder is 0.139 while a trained node's real loss "
                             "has a median of 0.0008, so 'select the highest I-value' became "
                             "'prefer whatever you have not seen'. 'optimistic' makes that "
                             "explicit, 'pessimistic' makes the estimator earn a visit, "
                             "'neutral' reproduces the historical behaviour.")
    parser.add_argument('--selection-diagnostic', action='store_true', default=False,
                        help="Write selection_diagnostic.csv.gz: per training batch, the mean "
                             "pairwise cosine distance between face embeddings plus label and "
                             "demographic composition. Measures whether a graph walk's batches "
                             "are less diverse than i.i.d. ones, which is the leading "
                             "explanation for i-value training losing to i.i.d. sampling.")
    parser.add_argument('--ivalue-diagnostic', action='store_true', default=False,
                        help="Write ivalue_diagnostic.csv.gz: predicted I-value against the "
                             "measured per-sample loss reduction, for checking whether the "
                             "estimator tracks learning gain at all. Costs one extra DQN "
                             "forward pass per trained sample.")
    parser.add_argument('--max-nodes-per-epoch', type=int, default=None,
                        help="Nodes trained on per epoch -- the run's real training "
                             "budget. --train-steps only bounds how far the traversal "
                             "walks. Defaults differ per capability (5000 basic, 10000 "
                             "DQN), so comparing an i-value arm against a random one "
                             "without setting this confounds sample selection with "
                             "sample count. Set it equal across arms.")
    parser.add_argument('--preprocess-workers', type=int, default=8,
                        help="Threads used to load and transform each DQN batch's images. "
                             "Only affects I-value traversals, which are the ones that "
                             "spend ~28%% of training in serial image loading at 6%% GPU "
                             "utilisation. Results are collected in submission order, so "
                             "this changes speed and not numbers; 1 forces the old serial "
                             "path.")
    parser.add_argument('--val-num-workers', type=int, default=4,
                        help='Number of parallel workers for validation image loading (default: 4)')

    # Uncertainty configuration
    parser.add_argument('--uncertainty-head', type=str, default='none',
                        choices=['none', 'evidential', 'batchensemble', 'sngp'],
                        help='Classifier head used to model predictive uncertainty (default: none)')
    parser.add_argument('--mc-dropout-samples', type=int, default=0,
                        help='Number of Monte Carlo dropout samples to use during evaluation (default: 0, disabled)')
    parser.add_argument('--batchensemble-members', type=int, default=4,
                        help='Number of BatchEnsemble members when --uncertainty-head=batchensemble (default: 4)')
    parser.add_argument('--sngp-hidden-dim', type=int, default=256,
                        help='Hidden dimension for the SNGP head (default: 256)')
    parser.add_argument('--sngp-rff-dim', type=int, default=256,
                        help='Random Fourier feature dimension for the SNGP head (default: 256)')
    parser.add_argument('--uncertainty-dropout-rate', type=float, default=0.2,
                        help='Dropout rate used by uncertainty-aware heads (default: 0.2)')
    parser.add_argument('--uncertainty-train-frequency', type=int, default=10,
                        help='Compute uncertainty summaries every N training batches (default: 10)')
    parser.add_argument('--sngp-precision-policy', type=str, default='per-epoch',
                        choices=['per-epoch', 'final-epoch', 'never-reset'],
                        help='When to reset the SNGP Laplace precision matrix. per-epoch (default) '
                             'makes gp_variance comparable between epochs; final-epoch matches the '
                             'original single-pass formulation; never-reset reproduces the pre-fix '
                             'behavior, where precision accumulated across all epochs')
    parser.add_argument('--graph-uncertainty-methods', type=str,
                        default='attribute_distance,embedding_distance,hybrid_distance,degree_penalty',
                        help='Comma-separated graph uncertainty methods. degree_penalty is the '
                             'ablation control: without it you cannot tell whether the distance '
                             'methods predict error or merely flag low-degree nodes '
                             '(default: all four)')
    parser.add_argument('--graph-degree-penalty-weight', type=float, default=1.0,
                        help='Strength of the low-degree uncertainty penalty (default: 1.0)')
    parser.add_argument('--graph-distance-robust-stats', action='store_true',
                        help='Standardize graph-distance attributes with median/IQR rather than '
                             'mean/std (default: enabled; blur is heavy-tailed)')
    parser.add_argument('--no-graph-distance-robust-stats', dest='graph_distance_robust_stats',
                        action='store_false', help='Use mean/std standardization instead')
    parser.set_defaults(graph_distance_robust_stats=True)
    parser.add_argument('--build-val-test-edges', action='store_true',
                        help='Build and cache validation/test graph edges the same way as training (default: enabled)')
    parser.add_argument('--no-build-val-test-edges', dest='build_val_test_edges', action='store_false',
                        help='Skip building validation/test edges and keep node-only graphs')
    parser.set_defaults(build_val_test_edges=True)

    # Model construction
    parser.add_argument('--finetune', action='store_true',
                        help='Freeze the backbone and train only the classifier head. NOTE: this is '
                             'what the detectors mean by "finetune" -- effnetdf and swintransformdf '
                             'freeze every parameter whose name lacks classifier/head. It was '
                             'previously hardcoded on, so runs on those architectures were linear '
                             'probes rather than fine-tuned detectors')
    parser.add_argument('--no-finetune', dest='finetune', action='store_false',
                        help='Train the whole network (default)')
    parser.set_defaults(finetune=False)

    # Benchmark artifacts
    parser.add_argument('--uq-records', action='store_true',
                        help='Write per-sample prediction records for the final test '
                             'evaluation to run_outputs/<run-id>/<description>/records.csv.gz. '
                             'This is the benchmark\'s only input: the metrics dict printed to '
                             'stdout carries batch means on incomparable scales, which cannot '
                             'answer which uncertainty method is better')
    parser.add_argument('--tune-threshold', action='store_true',
                        help='Fit the decision threshold on the val records and report '
                             'test at both 0.5 and the fitted point. Needs val in '
                             '--uq-records-splits. Temperature scaling cannot do this: '
                             'dividing a logit by T preserves its sign, so no prediction '
                             'moves across the boundary. Writes threshold_fit.json beside '
                             'the records; the record table itself is unchanged, since a '
                             'threshold is a decision rule and every ranking metric is '
                             'invariant to it')
    parser.add_argument('--threshold-objective', type=str, default='balanced_accuracy',
                        choices=['balanced_accuracy', 'youden_j', 'accuracy'],
                        help='What the fitted threshold maximizes on val. Do not use '
                             'accuracy on an imbalanced split: maximizing accuracy at ~87%% '
                             'prevalence pushes the threshold back toward predicting one '
                             'class, which is the behavior this flag exists to correct '
                             '(default: balanced_accuracy)')
    parser.add_argument('--uq-records-splits', type=str, default='test',
                        help='Comma-separated splits to record when --uq-records is set. '
                             'Temperature scaling has to be fitted on val and applied to test, '
                             'so a benchmark run needs both (default: test)')

    # Distribution shift
    parser.add_argument('--holdout', type=str, default='none',
                        help='Held-out source-group family. Removes those generators '
                             'from train and val and labels them domain=ood on test '
                             '(test nodes are never dropped, so a holdout run and its '
                             'control score the same samples). Holding out generators '
                             'shifts the class prior, so each holdout needs a paired '
                             '--holdout none control on the same reduced set. '
                             'See evaluation/uq/holdouts.py for the list')
    parser.add_argument('--list-holdouts', action='store_true',
                        help='Print the available holdouts with their measured sizes '
                             'and exit.')
    parser.add_argument('--corruption', type=str, default='none',
                        choices=['none', 'gaussian_blur', 'jpeg', 'gaussian_noise'],
                        help='Image corruption applied to the final test evaluation, '
                             'before the model resize. Severity 0 is byte-identical to '
                             'clean (default: none)')
    parser.add_argument('--corruption-severity', type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5],
                        help='Corruption severity. 0 is the identity (default: 0)')

    # Deep ensembles
    parser.add_argument('--ensemble-member', type=int, default=None,
                        help='Index of this run within a deep ensemble. Varies model '
                             'initialization while leaving --seed fixed, so all members share '
                             'one graph cache -- the graph cache key embeds the seed, so N '
                             'differently-seeded members would each rebuild the train graph. '
                             'It is also the better experiment: an ensemble should differ in '
                             'initialization, not in its training data')
    parser.add_argument('--ensemble-id', type=str, default=None,
                        help='Identifier shared by every member of one ensemble, recorded in '
                             'determinism.json so members can be discovered without globbing')

    # Reproducibility
    parser.add_argument('--determinism', type=str, default='fast', choices=['strict', 'fast'],
                        help='strict = bit-exact everywhere (deterministic algorithms, TF32 off, '
                             'single-threaded, AMP off, ordered collection); fast = perf-oriented, '
                             'GPU tolerance allowed (default: fast)')
    parser.add_argument('--strict-determinism', dest='determinism', action='store_const',
                        const='strict', help='Shorthand for --determinism strict')
    parser.add_argument('--lr-schedule', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'step', 'none'],
                        help='LR schedule. plateau branches on float comparisons, so a 1e-7 wobble '
                             'in val loss can flip an LR drop; cosine is a pure function of the '
                             'epoch index and therefore immune (default: plateau)')

    return parser.parse_args(argv)
