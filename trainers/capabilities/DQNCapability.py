import torch
import traceback
import torch.nn.functional as F
import random
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler
import os

#: Consecutive unpreprocessable batches before giving up rather than spinning silently.
MAX_CONSECUTIVE_PREPROCESS_FAILURES = 5

#: Consecutive whole-batch failures tolerated before the epoch gives up.
#:
#: The batch body is wrapped in `except Exception: continue`, and a failure there does not
#: advance `nodes_processed` -- so with `--train-steps 4000000` a fault that recurs every
#: batch spins forever instead of failing. Measured: a wrapped legacy estimator that raised
#: on every DQN update logged **258,398** identical tracebacks and held a GPU for nearly
#: three days without completing one epoch, while the run looked merely slow. Bounded, so the
#: process dies with the reason instead of quietly occupying hardware.
MAX_CONSECUTIVE_BATCH_FAILURES = 50

#: Ceiling on how much of the graph `--ivalue-ban-negative-gain` may withdraw, mirroring
#: `PerformanceGraphManager.MAX_REMOVAL_FRACTION`. 38% of trained samples show a negative
#: measured gain, so an unbounded ban could withdraw a third of the corpus and the arm would
#: then be measuring "train on less data", not "train on data that helps".
DEFAULT_BAN_MAX_FRACTION = 0.2

#: Threads used to read and decode one batch's images.
#:
#: `_preprocess_batch` was a serial Python loop doing a disk read, an image decode and a
#: torchvision transform per node while the GPU sat idle -- profiling a 3,000-step i-value
#: epoch put 21.9 s of 79.4 s (28%) in this function against 20.1 s of actual backward
#: pass, at 6% GPU utilisation. Of those 21.9 s, `ImageFileData.load_data` is 9.9 s and the
#: transform is the remaining ~12 s.
#:
#: Only the *load* is threaded, and that split is not an optimisation detail -- it is what
#: keeps the run reproducible. `CNNModel.train_transforms` is five random augmentations
#: (RandomHorizontalFlip, RandomRotation, ColorJitter, RandomAffine, RandomErasing) drawing
#: from the *global* torch RNG, so transforming on N threads interleaves those draws
#: non-deterministically: it changes which augmentation each image gets and the RNG state
#: every later consumer inherits. Measured, threading the transform too gave a different
#: record-table digest and a different node count (3,021 vs 3,002). Collecting results in
#: submission order is not sufficient protection -- ordering fixes the arrangement of the
#: results, not a shared mutable RNG.
#:
#: The load has no such state: `imread` is a pure function of the path and releases the GIL
#: in C, so threading it overlaps I/O and leaves the transform to run serially in the same
#: order as before. Bit-identical output, verified against `--preprocess-workers 1`.
DEFAULT_PREPROCESS_WORKERS = 8

#: How the DQN's reward -- and therefore the meaning of an I-value -- is defined.
#:
#: ``confidence`` (the original): reward is ``+confidence`` when the detector is right and
#: ``-confidence`` when it is wrong, so a high Q means "already mastered". `predict_i_value`
#: then returns ``1 - sigmoid(Q)``, which flips that into "the model does poorly here", and
#: the traversal maximising I-value is doing uncertainty sampling. The inversion is what
#: makes the sign work, and it is easy to misread in isolation.
#:
#: ``learning_gain``: reward is the sample's *measured* loss reduction across the update it
#: took part in. A high Q now already means "training on this helped", so the inversion in
#: `predict_i_value` must be undone -- see `get_i_value`. Getting that backwards yields a
#: confidently anti-informative sampler that trains without any error, which is why the
#: mapping is asserted in `tests/unit/test_ivalue_reward.py`.
IVALUE_REWARDS = ("confidence", "learning_gain")
DEFAULT_IVALUE_REWARD = "confidence"

from trainers.capabilities.BasicTrainingCapability import MAX_CONSECUTIVE_EMPTY_BATCHES
from trainers.capabilities.node_state import (
    DEFAULT_UNSEEN_PRIOR, NodeTrainingState, STATE_FEATURE_COUNT, STATE_LOSS_INDEX,
)
from trainers.capabilities.loss_weighting import DEFAULT_BAND, DEFAULT_WEIGHT_CLIP, LossWeighter
from trainers.capabilities.selection_diagnostic import SelectionDiagnostic
from trainers.capabilities.uncertainty_logging import (
    uncertainty_summary_for_logging as _uncertainty_summary_for_logging,
)
from models.DQNModel import DQNModel
from utils.attribute_utils import AttributeMetadata, AttributeBiasLoss
from nodes.atrnode import AttributeNode


def _stream(component, fallback_seed=0):
    """Private RNG stream for `component`.

    Replaces draws from the process-global `random` module. Sharing that global
    stream meant RNG consumption anywhere upstream shifted these decisions.
    """
    from test_helpers.determinism import component_rng
    return component_rng(component, fallback_seed=fallback_seed)


class DQNCapability:
    """Encapsulates all DQN-related functionality."""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.device = trainer.device
        self.attribute_metadata = trainer.attribute_metadata
        # Respect selected DQN model type on the trainer (default to 'basic')
        self.dqn_model_type = getattr(trainer, 'dqn_model_type', 'basic')
        print(f"DQNCapability: Using DQN model type '{self.dqn_model_type}'")
        
        # DQN settings
        self.embedding_dim = 512
        self.feature_dim = None
        self.dqns = []
        
        # Image loading threads for `_preprocess_batch`. Off the trainer so one flag
        # reaches every capability, and floored at 1 so `--preprocess-workers 0` means
        # "serial" rather than crashing a ThreadPoolExecutor.
        self.preprocess_workers = max(
            1, int(getattr(trainer, 'preprocess_workers', DEFAULT_PREPROCESS_WORKERS) or 1)
        )

        # I-value definition. Off-by-default so an existing command line reproduces exactly.
        reward = getattr(trainer, 'ivalue_reward', None) or DEFAULT_IVALUE_REWARD
        if reward not in IVALUE_REWARDS:
            raise ValueError(
                f"unknown ivalue_reward {reward!r}; choose from {', '.join(IVALUE_REWARDS)}"
            )
        self.ivalue_reward = reward
        if int(getattr(trainer, 'ivalue_ban_negative_gain', 0) or 0) and reward != 'learning_gain':
            raise ValueError(
                "--ivalue-ban-negative-gain needs a measured gain per sample, which only "
                "--ivalue-reward learning_gain produces (it is the extra post-update forward "
                "pass). Pass both, or neither."
            )
        self.use_state_features = bool(getattr(trainer, 'ivalue_state_features', False))
        self.dqn_fixes = bool(getattr(trainer, 'dqn_fixes', False))
        self.dqn_objective = getattr(trainer, 'dqn_objective', None) or 'rank'
        self.dqn_buffer_size = int(getattr(trainer, 'dqn_buffer_size', None) or 512)
        # Banning: withdraw nodes whose mean *measured* gain stays negative. 0 disables.
        self.ban_min_visits = int(getattr(trainer, 'ivalue_ban_negative_gain', 0) or 0)
        self.ban_max_fraction = float(
            getattr(trainer, 'ivalue_ban_max_fraction', None) or DEFAULT_BAN_MAX_FRACTION
        )
        self.banned_nodes = set()
        self.bans_this_epoch = 0
        # Same weighter the basic path uses. Needed here because
        # `CapabilityManager.train_with_traversal` routes *every* traversal through this path
        # once a DQN exists -- including `comprehensive`, which is exactly the sampler the
        # weighting arm wants. Implemented only in the basic path, it was dead code.
        self.weighter = LossWeighter(
            mode=getattr(trainer, 'ivalue_loss_weight', None) or 'none',
            clip=getattr(trainer, 'ivalue_weight_clip', None) or DEFAULT_WEIGHT_CLIP,
            band=getattr(trainer, 'ivalue_selection_band', None) or DEFAULT_BAND,
        )
        self.unseen_prior = getattr(trainer, 'ivalue_unseen_prior', None) or DEFAULT_UNSEEN_PRIOR
        self.node_state = (
            NodeTrainingState(unseen_prior=self.unseen_prior)
            if self.use_state_features else None
        )
        #: Records what each batch actually contained. Shared with the basic path via the
        #: trainer, so a run that switches traversals produces one comparable file.
        self.selection_diagnostic = getattr(trainer, 'selection_diagnostic', None)
        #: Filled when a diagnostic sink is attached; one row per trained sample.
        self.ivalue_diagnostic_rows = []
        self.collect_ivalue_diagnostic = bool(
            getattr(trainer, 'collect_ivalue_diagnostic', False)
        )
        self.current_epoch = 0

        # Training settings
        self.batch_size = 32
        self.gradient_accumulation_steps = 8
        # See BasicTrainingCapability: the two defaults differ (10000 vs 5000), so this
        # is the knob that makes an i-value arm and a random arm comparable.
        self.max_nodes_per_epoch = int(
            getattr(trainer, 'max_nodes_per_epoch', None) or 10000
        )
        self.scaler = GradScaler()
        
        # Prediction stats for bias tracking
        self.prediction_stats = defaultdict(lambda: defaultdict(list))
        
        # Initialize DQN models if attribute metadata exists
        if self.attribute_metadata:
            self._initialize_dqns()
        else:
            print("DQNCapability: No attribute metadata provided. DQN not initialized.")
            
    def _initialize_dqns(self):
        """Initialize DQN models based on attribute metadata and selected model type."""
        try:
            # Get a sample node to determine feature dimensions
            sample_nodes = list(self.trainer.graphmanager.get_graph().get_nodes())
            if not sample_nodes:
                raise ValueError("No nodes available for DQN initialization")
                
            sample_node = sample_nodes[0]
            
            # Calculate feature dimensions
            features_tensor, embedding_tensor = self._get_dqn_features(sample_node)
            if features_tensor is None:
                raise ValueError("Could not extract features from sample node")
                
            self.feature_dim = features_tensor.shape[0]
            print(f"DQNCapability: Calculated feature dimension: {self.feature_dim}")
            
            # Initialize DQN for each model. One factory, shared with
            # `test_hierarchical.create_dqn_model`: this dispatch used to be duplicated, which
            # is how a `--dqn-model gain_*` name could be accepted by the CLI and still never
            # reach the capability.
            from models.gain_estimator import build_estimator

            model_type = (self.dqn_model_type or 'basic').lower()
            for i, _model in enumerate(self.trainer.models):
                dqn = build_estimator(
                    model_type,
                    feature_dim=self.feature_dim,
                    device=self.device,
                    embedding_dim=self.embedding_dim,
                    apply_fixes=self.dqn_fixes,
                    state_dim=STATE_FEATURE_COUNT if self.use_state_features else 0,
                    objective=self.dqn_objective,
                    buffer_size=self.dqn_buffer_size,
                )
                self.dqns.append(dqn)
                print(
                    f"DQNCapability: Initialized estimator {i} (type={model_type}, "
                    f"fixes={self.dqn_fixes}, objective={self.dqn_objective}, "
                    f"feature_dim={self.feature_dim}, "
                    f"state_dim={STATE_FEATURE_COUNT if self.use_state_features else 0})"
                )

        except Exception as e:
            print(f"DQNCapability: Error initializing DQN: {e}. DQN not initialized.")
            self.dqns = []
            
    def _get_dqn_features(self, node):
        """Extract attribute features for DQN input."""
        try:
            if not isinstance(node, AttributeNode):
                print(f"Warning: Expected AttributeNode, got {type(node)}. Cannot extract DQN features.")
                return None, None

            features_list = []
            embedding_data = None

            # Extract standard attributes based on metadata
            if self.attribute_metadata:
                for attr_meta in self.attribute_metadata:
                    attr_name = attr_meta['name'] if isinstance(attr_meta, dict) else attr_meta.name
                    attr_type = attr_meta['type'] if isinstance(attr_meta, dict) else attr_meta.attr_type

                    # Special handling for face embedding
                    if attr_name == 'face_embedding':
                        embedding_data = node.attributes.get(attr_name)
                        continue

                    # Handle other attributes
                    if attr_type == 'categorical':
                        # One-hot encode categorical values
                        possible_values = (attr_meta.get('possible_values') if isinstance(attr_meta, dict) 
                                         else attr_meta.possible_values)
                        if possible_values:
                            for possible_value in possible_values:
                                features_list.append(1.0 if node.attributes.get(attr_name) == possible_value else 0.0)
                    else:  # continuous
                        try:
                            features_list.append(float(node.attributes.get(attr_name, 0)))
                        except (TypeError, ValueError) as e:
                            print(f"Warning: Could not convert attribute '{attr_name}' value '{node.attributes.get(attr_name)}' to float: {e}. Using 0.")
                            features_list.append(0.0)
            elif isinstance(node, AttributeNode):
                # Fallback: Use default features if no metadata
                features_list.append(float(node.label))  # Label (0 or 1)
                features_list.append(len(node.get_adjacent_nodes()) / 100.0)  # Normalized degree

            # Model-state features, so the ranking can change as the model learns. Appended
            # unconditionally when enabled (neutral values for an unseen node) because
            # `_initialize_dqns` probes `feature_dim` once -- a variable length here would
            # size the input layer off whichever node happened to be sampled first.
            if self.use_state_features and self.node_state is not None:
                features_list.extend(
                    self.node_state.features(node, epoch=self.current_epoch)
                )

            # Convert features list to tensor
            try:
                features_tensor = torch.tensor(features_list, dtype=torch.float32)
            except Exception as e:
                print(f"Error converting features list to tensor: {e}. List: {features_list}")
                features_tensor = None

            # Handle embedding: convert to tensor, then pad/truncate to expected dim
            embedding_tensor = None
            if embedding_data is not None:
                try:
                    # Convert and flatten if needed
                    embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)
                    if embedding_tensor.ndim > 1:
                        embedding_tensor = embedding_tensor.flatten()
                    # Ensure fixed length equal to self.embedding_dim
                    current_len = int(embedding_tensor.numel())
                    target_len = int(self.embedding_dim)
                    if target_len > 0 and current_len != target_len:
                        if current_len > target_len:
                            embedding_tensor = embedding_tensor[:target_len]
                        else:
                            pad_len = target_len - current_len
                            # Ensure padding tensor is on the same device as embedding_tensor
                            embedding_tensor = torch.cat([
                                embedding_tensor,
                                torch.zeros(pad_len, dtype=torch.float32, device=embedding_tensor.device)
                            ], dim=0)
                except Exception as e:
                    print(f"Error converting/padding embedding to tensor: {e}. Using zeros of len {self.embedding_dim}.")
                    embedding_tensor = torch.zeros(self.embedding_dim, dtype=torch.float32)

            return features_tensor, embedding_tensor

        except Exception as e:
            print(f"Error extracting DQN features: {e}")
            return None, None
            
    def get_i_value(self, node, model_idx=0):
        """Calculate I-value using DQN."""
        try:
            # Check if DQN models are available
            if not self.dqns or model_idx >= len(self.dqns):
                print(f"Warning: DQN model {model_idx} not available for I-value calculation.")
                return 0.0

            # Get the target DQN model and its device
            dqn_model = self.dqns[model_idx]
            target_device = dqn_model.device

            # Retrieve features and embedding for the node
            features_tensor, embedding_tensor = self._get_dqn_features(node)

            # Check if feature extraction was successful
            if features_tensor is None:
                print(f"Warning: Could not extract features for node {node.node_id}. Cannot calculate I-value.")
                return 0.0

            # Handle potentially missing embedding tensor
            if embedding_tensor is None:
                embedding_tensor = torch.zeros(self.embedding_dim, device=target_device)
            elif not isinstance(embedding_tensor, torch.Tensor):
                try:
                    embedding_tensor = torch.tensor(embedding_tensor, dtype=torch.float32, device=target_device)
                except Exception as e:
                    print(f"Error converting embedding to tensor for node {node.node_id}: {e}. Using zeros.")
                    embedding_tensor = torch.zeros(self.embedding_dim, device=target_device)
            else:
                embedding_tensor = embedding_tensor.to(target_device)

            # Features should already be a tensor - ensure it's on the correct device
            features_tensor = features_tensor.to(target_device)

            # Ensure DQN model is on the correct device (move all parameters)
            if next(dqn_model.parameters()).device != target_device:
                dqn_model = dqn_model.to(target_device)
                # Update the reference in the list
                self.dqns[model_idx] = dqn_model

            # Add batch dimension
            features_tensor = features_tensor.unsqueeze(0)
            embedding_tensor = embedding_tensor.unsqueeze(0)

            # Get I-value from DQN (predict_i_value will handle device placement internally)
            i_value = dqn_model.predict_i_value(features_tensor, embedding_tensor)

            i_value = i_value.detach().cpu().item()

            # Which way does this number point? Ask the model, not the reward.
            #
            # The legacy family returns `1 - sigmoid(Q)`. That is right for the confidence
            # reward -- a high Q means already-mastered, so the inversion turns it into "the
            # model does poorly here" and maximising it samples hard examples -- but wrong
            # under the learning-gain reward, where a high Q already means "training on this
            # helped". The fixed estimators return a raw score that already points at
            # informativeness and must not be touched. Reading `value_semantics` puts that
            # knowledge in the one place that actually has it: the model.
            if getattr(dqn_model, 'value_semantics', 'legacy_inverted') != 'informativeness':
                if self.ivalue_reward == "learning_gain":
                    i_value = 1.0 - i_value

            # Deliberately not `i_value > 0.5`, which is what this used to pass: that records
            # a node as "correct" exactly when the DQN says the model handles it badly.
            predicted_correct = i_value < 0.5
            self.update_prediction_stats(node, predicted_correct, model_idx)

            return i_value
            
        except Exception as e:
            print(f"Error calculating I-value: {str(e)}")
            return 0.0
            
    def update_prediction_stats(self, node, correct, model_idx):
        """Update prediction statistics for each attribute value."""
        if not self.attribute_metadata:
            return
            
        node_attrs = node.attributes
        
        for attr in self.attribute_metadata:
            attr_name = attr['name'] if isinstance(attr, dict) else attr.name
            if attr_name in node_attrs:
                value = node_attrs[attr_name]
                attr_type = attr['type'] if isinstance(attr, dict) else attr.attr_type
                if attr_type == 'categorical':
                    self.prediction_stats[f'model_{model_idx}_{attr_name}'][value].append(float(correct))
                    
    def train_with_dqn(self, traversal, epoch=None):
        """Training loop with DQN integration."""
        try:
            # Set models to training mode
            for model in self.trainer.models:
                model.train()
            for dqn in self.dqns:
                dqn.train()
                
            # Initialize metrics
            total_loss = 0.0
            correct = 0
            total = 0
            batch_count = 0
            batches_failed = 0
            total_train_bias_loss = 0.0
            uncertainty_sums = defaultdict(float)
            uncertainty_counts = defaultdict(int)

            # Before resetting: the staleness feature is read during selection, so the epoch
            # has to be current from the first traverse() call, not from the first DQN update.
            self.current_epoch = int(epoch or 0)

            # Reset traversal for this epoch
            traversal.reset_pointers()
            
            # Get total nodes for this epoch
            total_nodes = traversal.num_steps
            print(f"Training on {total_nodes} nodes this epoch with DQN")
            
            # Track attribute distribution
            attribute_distribution = defaultdict(lambda: defaultdict(int))
            track_attributes = bool(self.trainer.categorical_attrs_for_tracking)
            
            nodes_processed = 0
            empty_batches = 0
            batches_errored = 0
            pbar = tqdm(total=min(total_nodes, self.max_nodes_per_epoch), desc="DQN Training")
            
            while nodes_processed < min(total_nodes, self.max_nodes_per_epoch):
                try:
                    # Get batch from traversal. An empty batch is a local dead end, not
                    # an exhausted graph, and the same tolerance the basic path uses
                    # applies here -- both arms must give up at the same point or their
                    # realised sample counts differ.
                    batch_nodes = traversal.traverse(self.batch_size)
                    if not batch_nodes:
                        empty_batches += 1
                        if empty_batches >= MAX_CONSECUTIVE_EMPTY_BATCHES:
                            break
                        continue
                    empty_batches = 0

                    batch_nodes = self._drop_harmful(batch_nodes)
                    if not batch_nodes:
                        continue

                    # Never overshoot the budget: the arms must match exactly, not roughly.
                    remaining = min(total_nodes, self.max_nodes_per_epoch) - nodes_processed
                    if len(batch_nodes) > remaining:
                        batch_nodes = batch_nodes[:remaining]
                        
                    # Preprocess batch. A failure here does not advance `nodes_processed`,
                    # so the loop would spin until the traversal exhausted its steps and then
                    # report a perfectly successful epoch with `avg_loss: 0.0` -- which is
                    # how a hardcoded `.cuda()` hid for as long as it did. Counted, and
                    # raised on if nothing at all can be preprocessed.
                    images, batch_nodes_loaded = self._preprocess_batch(batch_nodes)
                    if images is None or not batch_nodes_loaded:
                        batches_failed += 1
                        if batches_failed >= MAX_CONSECUTIVE_PREPROCESS_FAILURES:
                            raise RuntimeError(
                                f"{batches_failed} consecutive batches could not be "
                                f"preprocessed, so no training step has run. The first "
                                f"error is printed above; a device mismatch or an "
                                f"unreadable image is the usual cause."
                            )
                        continue
                    batches_failed = 0
                        
                    # Extract labels
                    batch_labels_loaded = [float(node.get_label()) for node in batch_nodes_loaded]
                    batch_labels_tensor = torch.tensor(batch_labels_loaded, dtype=torch.float).unsqueeze(1).to(self.device)
                    
                    # Forward pass
                    model = self.trainer.models[0]
                    if hasattr(model, 'forward_with_uncertainty'):
                        summarize_now = (
                            batch_count % max(1, model.uncertainty_train_frequency) == 0
                        )
                        # Single deterministic pass for the loss; MC statistics are
                        # gathered separately under no_grad for logging. Running MC
                        # dropout in the loss path every Nth batch made the training
                        # objective non-stationary.
                        prediction_bundle = model.forward_with_uncertainty(
                            images,
                            nodes=batch_nodes_loaded,
                            update_precision=True,
                            use_mc_dropout=False,
                            compute_variance=summarize_now,
                        )
                        outputs = prediction_bundle.logits
                        loss = model.compute_loss(
                            prediction_bundle,
                            batch_labels_tensor,
                            base_criterion=self.trainer.criterion,
                        )
                        if summarize_now:
                            for name, value in _uncertainty_summary_for_logging(
                                model, prediction_bundle, images, batch_nodes_loaded
                            ).items():
                                uncertainty_sums[name] += float(value) * len(batch_nodes_loaded)
                                uncertainty_counts[name] += len(batch_nodes_loaded)
                    else:
                        outputs = model(images)
                        loss = self.trainer.criterion(outputs, batch_labels_tensor)
                    # Re-weight by I-value.
                    #
                    # Gated on the *head type*, not on which branch produced the loss: the
                    # first version hooked the plain-`criterion` else-branch, which `CNNModel`
                    # never takes because it always defines `forward_with_uncertainty`. The
                    # weighting was therefore silently inert and the arm would have run as its
                    # own control. With `uncertainty_head=none`, `compute_loss` is a plain BCE
                    # and swapping in the weighted per-sample mean is exact; the evidential and
                    # batchensemble heads replace or reshape the loss, so they are excluded.
                    #
                    # `_per_sample_bce` defaults to `detach=True` because its other two
                    # call sites (the DQN's own reward diagnostic) must never carry gradient.
                    # This call site is different: its result *replaces* `loss` below and is
                    # what actually reaches `.backward()`. Detaching here silently zeroed the
                    # classifier's gradient every batch -- and it did not crash, because
                    # `total_loss_for_backward = loss + bias_weight * bias_loss_val` stayed
                    # in the autograd graph (and `.backward()` succeeded) purely via the bias
                    # term, even though `bias_weight` was correctly zero: multiplying a live
                    # tensor by a zero *coefficient* does not detach it, it just zeroes that
                    # term's contribution. So every batch ran an optimizer step on an
                    # all-zero gradient, and the only thing that moved the weights at all was
                    # AdamW's decoupled weight decay -- which explains both the chance-level
                    # AUROC and why it drifted slightly rather than sitting frozen.
                    head_type = getattr(self.trainer.models[0], 'uncertainty_head_type', None)
                    if self.weighter.enabled and head_type in (None, 'none'):
                        per_sample = self._per_sample_bce(outputs, batch_labels_loaded, detach=False)
                        values = []
                        for node in batch_nodes_loaded:
                            try:
                                values.append(float(self.get_i_value(node, 0)))
                            except Exception:
                                values.append(float('nan'))
                        loss = self.weighter.apply(per_sample, values)

                    # Add bias loss if available
                    bias_loss_val = 0.0
                    bias_weight = 0.0
                    bias_loss_fn = self.trainer.capabilities.get_bias_loss()
                    if bias_loss_fn:
                        try:
                            bias_loss_val = bias_loss_fn(outputs, batch_labels_tensor, batch_nodes_loaded)
                            bias_weight = getattr(self.trainer.capabilities.bias_capability, 'bias_weight', 0.0)
                            total_train_bias_loss += bias_loss_val.item()
                        except Exception as e:
                            print(f"Warning: Error calculating bias loss: {e}")
                    # Combine losses
                    total_loss_for_backward = loss + bias_weight * bias_loss_val
                    # Calculate metrics
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    correct += (preds == batch_labels_tensor).sum().item()
                    total_loss += loss.item()
                    total += len(batch_labels_loaded)
                    # Track attribute distribution
                    if track_attributes:
                        for node in batch_nodes_loaded:
                            for attr_name in self.trainer.categorical_attrs_for_tracking:
                                if attr_name in node.attributes:
                                    attr_value = node.attributes[attr_name]
                                    attribute_distribution[attr_name][attr_value] += 1
                    # Backward pass
                    self.scaler.scale(total_loss_for_backward).backward()
                    self.scaler.step(self.trainer.models[0].optim)
                    self.scaler.update()
                    self.trainer.models[0].optim.zero_grad()
                    
                    # DQN Training Integration
                    if self.dqns:
                        self._train_dqn_on_batch(
                            batch_nodes_loaded, outputs, batch_labels_loaded,
                            images=images, epoch=epoch,
                        )
                    
                    batches_errored = 0
                    batch_count += 1
                    nodes_processed += len(batch_nodes)
                    pbar.update(len(batch_nodes))
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                        
                except Exception as e:
                    # Traceback, not just the message. A bare `str(e)` here turns any
                    # per-batch fault into an unlocatable one-liner repeated thousands of
                    # times, and because the loop continues the epoch still reports metrics
                    # -- so the run looks healthy while training on a fraction of its batches.
                    batches_errored += 1
                    print(f"Error processing batch: {str(e)}")
                    traceback.print_exc()
                    if batches_errored >= MAX_CONSECUTIVE_BATCH_FAILURES:
                        raise RuntimeError(
                            f"{batches_errored} consecutive batches failed, so this epoch has "
                            f"made no progress. A failure here does not advance "
                            f"`nodes_processed`, so continuing would spin until the traversal "
                            f"exhausts its step budget -- which at --train-steps 4000000 is "
                            f"effectively never. The first traceback above is the cause."
                        ) from e
                    continue
                    
            pbar.close()
            summary = self.weighter.summary_and_reset()
            if summary:
                print(f"  loss weighting: mode={self.weighter.mode} "
                      f"mean weight={summary[0]:.4f} over {summary[1]} sample(s)")
            if self.ban_min_visits:
                print(f"  banning: {self.bans_this_epoch} node(s) withdrawn this epoch, "
                      f"{len(self.banned_nodes)} total")
                self.bans_this_epoch = 0
            
            # Compute epoch metrics
            if batch_count == 0:
                return self._get_empty_metrics()
                
            metrics = {
                'avg_loss': total_loss / batch_count,
                'accuracy': correct / max(1, total),
                'avg_bias_loss': total_train_bias_loss / batch_count if total_train_bias_loss > 0 else 0.0
            }
            if uncertainty_sums:
                metrics['uncertainty_summary'] = {
                    name: uncertainty_sums[name] / max(1, uncertainty_counts[name])
                    for name in uncertainty_sums
                }
            
            return metrics, attribute_distribution
            
        except Exception as e:
            print(f"Error in DQN training: {str(e)}")
            return self._get_empty_metrics()
            
    def _preprocess_batch(self, batch_nodes):
        """Preprocess a batch of nodes to ensure consistent tensor sizes."""
        if not batch_nodes:
            return None, None
        
        try:
            # Get CNN model for transforms
            cnn_model = None
            for model in self.trainer.models:
                if hasattr(model, 'transform'):
                    cnn_model = model
                    break
                    
            if cnn_model is None:
                print("No model with transform method found")
                return None, None
                
            # Stage 1: read and decode. Pure function of the path, no shared state, so
            # it is safe to overlap across threads.
            def read_one(node):
                """Return decoded image data for `node`, or None to drop it."""
                try:
                    if not isinstance(node, AttributeNode):
                        return None
                    data = node.get_data()
                    if data is None:
                        return None
                    return data.load_data()
                except Exception as e:
                    print(f"Error processing node in batch: {str(e)}")
                    return None

            # getattr, not self.preprocess_workers: `_preprocess_batch` is documented
            # to need only `trainer.models[0].transform` and `self.device`, and is
            # built via `__new__` in tests to hold that line. Adding a hard __init__
            # dependency here broke it, so the default lives at the use site too.
            configured = getattr(self, 'preprocess_workers', DEFAULT_PREPROCESS_WORKERS)
            workers = max(1, min(int(configured or 1), len(batch_nodes)))
            if workers == 1:
                decoded = [read_one(node) for node in batch_nodes]
            else:
                # `map` yields in submission order, so stage 2 sees exactly the sequence
                # the serial loop saw -- which is what keeps the RNG draws below identical.
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    decoded = list(pool.map(read_one, batch_nodes))

            # Stage 2: transform. Serial on purpose -- the training transforms draw from
            # the global torch RNG, so threading this would change the augmentations and
            # the RNG state, not just the wall clock.
            processed_batch = []
            valid_nodes = []
            for node, img_data in zip(batch_nodes, decoded):
                if img_data is None:
                    continue
                if not isinstance(img_data, torch.Tensor):
                    try:
                        transform = self.trainer.models[0].transform
                        img_data = transform(img_data)
                    except Exception as e:
                        print(f"Error transforming image: {str(e)}")
                        continue
                processed_batch.append(img_data)
                valid_nodes.append(node)
                    
            if not processed_batch:
                return None, None
                
            # Stack tensors. `.to(self.device)`, not `.cuda()`: the hardcoded call made this
            # whole training path GPU-only. On a CPU run it raised, the caller's `continue`
            # swallowed it, and the epoch reported `avg_loss: 0.0` having trained on nothing
            # -- so `--traversal-type i-value` silently did not train at all without a GPU,
            # including under the strict determinism that pins CUDA_VISIBLE_DEVICES.
            try:
                images = torch.stack(processed_batch).to(self.device)
                return images, valid_nodes
            except Exception as e:
                print(f"Error stacking tensors: {str(e)}")
                return None, None
                
        except Exception as e:
            print(f"Error in preprocess_batch: {str(e)}")
            return None, None
            
    def _drop_harmful(self, batch_nodes):
        """Withdraw nodes whose mean measured gain has stayed negative.

        Capped at `ban_max_fraction` of the graph: past that the arm stops measuring "train
        on data that helps" and starts measuring "train on less data", which is a different
        and much less interesting experiment.
        """
        if not self.ban_min_visits or self.node_state is None:
            return batch_nodes

        graph = None
        try:
            graph = self.trainer.graphmanager.get_graph()
        except Exception:
            pass
        ceiling = (
            int(self.ban_max_fraction * len(graph.get_nodes())) if graph is not None else None
        )

        kept = []
        for node in batch_nodes:
            node_id = getattr(node, 'node_id', None)
            if node_id in self.banned_nodes:
                continue
            if self.node_state.is_harmful(node, min_visits=self.ban_min_visits):
                if ceiling is None or len(self.banned_nodes) < ceiling:
                    self.banned_nodes.add(node_id)
                    self.bans_this_epoch += 1
                    continue
            kept.append(node)
        return kept

    def _per_sample_bce(self, logits, labels, detach=True):
        """Per-sample binary cross-entropy from raw logits.

        Deliberately not `model.compute_loss`, which folds in the bias penalty and any
        uncertainty-head terms. Those are part of the training objective but they are not
        "how much did this sample teach the model", so including them would make the reward
        a different quantity from the one it claims to measure.

        `detach=True` is correct at both existing call sites (`_train_dqn_on_batch`'s
        `loss_before`/`loss_after`): they are pure measurement, computed to feed the DQN's
        reward, and must never carry gradient back into the classifier. It is **wrong** when
        the result is going to be used as the actual training loss -- which is exactly what
        the `ivalue_loss_weight` path in `train_with_dqn` does. That call passes
        `detach=False`; see the comment there for what the bug looked like.
        """
        logits = (logits.detach() if detach else logits).reshape(-1).float()
        target = torch.as_tensor(
            labels, dtype=torch.float32, device=logits.device
        ).reshape(-1)
        return F.binary_cross_entropy_with_logits(logits, target, reduction='none')

    def _train_dqn_on_batch(self, batch_nodes_loaded, outputs, batch_labels_loaded,
                            images=None, epoch=0):
        """Train DQN models on a batch of nodes.

        Called *after* `scaler.step()`, so `outputs` are the pre-update logits and the model
        has already moved. That ordering is what makes a measured learning gain available:
        one extra forward pass on the same images gives the post-update loss, and the
        difference is what training on this batch actually bought per sample.
        """
        if not self.dqns:
            return

        dqn_model = self.dqns[0]  # Use first DQN for now
        self.current_epoch = int(epoch)

        loss_before = self._per_sample_bce(outputs, batch_labels_loaded)
        if self.selection_diagnostic is not None:
            self.selection_diagnostic.record(
                batch_nodes_loaded, epoch=epoch,
                losses=loss_before.detach().cpu().tolist(), selector='i-value',
            )
        loss_after = None
        if self.ivalue_reward == "learning_gain":
            if images is None:
                raise ValueError(
                    "ivalue_reward='learning_gain' needs the batch's images to measure the "
                    "post-update loss; _train_dqn_on_batch was called without them."
                )
            model = self.trainer.models[0]
            # `CNNModel` is a wrapper, not an `nn.Module`: it has no `.training`, and its
            # `train()`/`eval()` drive its own `current_mode`, which also selects the image
            # transform. Restore whatever mode was active rather than assuming train, so a
            # measurement never leaves the model in the wrong mode for the next batch.
            previous_mode = getattr(model, 'current_mode', None)
            model.eval()
            try:
                with torch.no_grad():
                    post = model(images)
                    if not isinstance(post, torch.Tensor):
                        post = getattr(post, 'logits', post)
            finally:
                if previous_mode == 'eval':
                    model.eval()
                else:
                    model.train()
            loss_after = self._per_sample_bce(post, batch_labels_loaded)

        for i, node in enumerate(batch_nodes_loaded):
            # Calculate reward for DQN
            prediction_probability = torch.sigmoid(outputs[i]).item()
            is_correct = (prediction_probability > 0.5) == (batch_labels_loaded[i] > 0.5)

            # Calculate confidence-based reward
            confidence = abs(prediction_probability - 0.5) * 2
            reward_sign = 1.0 if is_correct else -1.0
            confidence_reward = reward_sign * confidence

            before = float(loss_before[i])
            if self.ivalue_reward == "learning_gain":
                after = float(loss_after[i])
                dqn_reward = before - after
            else:
                after = float('nan')
                dqn_reward = confidence_reward

            # Read the I-value *before* recording this observation. The DQN has not been
            # updated yet at this point in the function, so this reproduces exactly what the
            # traversal saw when it chose this node -- which is the quantity the gate needs
            # to correlate against the realised gain. Recording state first would leak the
            # outcome into the prediction and make the correlation look better than it is.
            predicted_ivalue = (
                self.get_i_value(node, 0) if self.collect_ivalue_diagnostic else None
            )
            # The loss the *estimator* could see when it chose this node: an EWMA of losses
            # from earlier visits, or a neutral constant if the node had never been trained
            # on. Recorded separately from `loss_before` because the two are not the same
            # quantity and conflating them flatters every estimator -- `loss_before` is
            # measured by the forward pass on this very batch, so it is not available at
            # selection time without paying that pass for every candidate.
            state_loss = None
            if self.collect_ivalue_diagnostic and self.node_state is not None:
                state_loss = self.node_state.features(
                    node, epoch=self.current_epoch
                )[STATE_LOSS_INDEX]

            # Record what the model currently does with this node, for the state features.
            # Uses the pre-update prediction: that is the state the selector saw.
            if self.node_state is not None:
                self.node_state.observe(node, prediction_probability, before, epoch=epoch)
                if self.ivalue_reward == "learning_gain":
                    self.node_state.observe_gain(node, before - after)

            if self.collect_ivalue_diagnostic:
                self.ivalue_diagnostic_rows.append({
                    'epoch': int(epoch),
                    'node_id': getattr(node, 'node_id', None),
                    'predicted_ivalue': predicted_ivalue,
                    'reward': dqn_reward,
                    'loss_before': before,
                    'loss_after': after,
                    'gain': before - after,
                    'prob_before': prediction_probability,
                    'correct_before': int(bool(is_correct)),
                    'state_loss': state_loss,
                })

            # Get DQN state
            dqn_features, dqn_embedding = self._get_dqn_features(node)

            if dqn_features is None:
                continue
        
            # Move features to DQN device
            dqn_features = dqn_features.to(dqn_model.device) 
            if dqn_embedding is not None:
                dqn_embedding = dqn_embedding.to(dqn_model.device)
        
            # Push experience to the buffer. `observe` lets the model timestamp it, which
            # is what makes recency weighting and an age cutoff possible at all.
            if hasattr(dqn_model, 'observe'):
                dqn_model.observe(
                    dqn_features.detach(),
                    dqn_embedding.detach() if dqn_embedding is not None else None,
                    dqn_reward,
                )
                continue
            dqn_model.replay_buffer.append((
                dqn_features.detach(), 
                dqn_embedding.detach() if dqn_embedding is not None else None, 
                dqn_reward
            ))

        # Perform a learning step. `observe`/`learn` means the model owns its buffer, which
        # is where the staleness decision has to live: sampling used to happen *here*, so a
        # model could shrink its own `maxlen` and still be handed a uniform draw across a
        # whole epoch of rewards belonging to a model that no longer exists.
        if hasattr(dqn_model, 'learn'):
            dqn_model.learn(rng=_stream('dqn.replay'))
        elif len(dqn_model.replay_buffer) >= dqn_model.batch_size:
            dqn_transitions = _stream('dqn.replay').sample(dqn_model.replay_buffer, dqn_model.batch_size)
            dqn_model.train_step(dqn_transitions)
            
    def _get_empty_metrics(self):
        """Return empty metrics structure for when no valid data is processed."""
        return {
            'avg_loss': 0.0,
            'accuracy': 0.0,
            'avg_bias_loss': 0.0
        }
    
    def save_checkpoint(self, checkpoint_path):
        """Save DQN models to checkpoint."""
        try:
            if self.dqns:
                # For now, save the first DQN model
                # In the future, we could save all DQN models
                self.dqns[0].save_checkpoint(checkpoint_path)
                print(f"DQN checkpoint saved to {checkpoint_path}")
                return True
            else:
                print("No DQN models to save")
                return False
        except Exception as e:
            print(f"Error saving DQN checkpoint: {e}")
            return False
            
    def load_checkpoint(self, checkpoint_path):
        """Load DQN models from checkpoint."""
        try:
            if self.dqns:
                # Check if file exists before attempting load
                if not os.path.exists(checkpoint_path):
                    print(f"Warning: DQN Checkpoint file not found at {checkpoint_path}. Skipping load.")
                    return False
                    
                # Load into the first DQN model
                self.dqns[0].load_checkpoint(checkpoint_path)
                print(f"DQN checkpoint loaded from {checkpoint_path}")
                return True
            else:
                print("No DQN models to load checkpoint into")
                return False
        except Exception as e:
            print(f"Error loading DQN checkpoint: {e}")
            return False 
