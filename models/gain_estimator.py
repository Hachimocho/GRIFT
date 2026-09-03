"""Estimators for a sample's expected learning gain, and the fixes they exist to apply.

Why this module exists
----------------------
The predecessor family (`DQNModel` plus the four in `EnhancedDQNModels`) was measured
against the thing it claims to predict and lost to a free baseline by 33x. On 20,000 paired
observations (`docs/ivalue_gate_result.md`) the predicted I-value ranked realised per-sample
learning gain at Spearman **+0.010**, while a sample's **current loss** ranked it at
**+0.331** -- and current loss was already one of the network's inputs, yet its output was
uncorrelated with it (-0.009). Three measured causes, each addressed here once rather than
five times:

1. **The informative features were swamped.** A 64-d compressed face embedding was
   concatenated with a 31-d feature vector, so the first layer saw 95 dims of which the 6
   model-state features were 6.3% and the embedding was 67%. The embedding is a static
   property of the image; it says nothing about what the detector currently knows.
2. **The replay buffer served stale labels for a non-stationary target.** `maxlen=10000` at
   10,000 samples an epoch holds a full epoch, but a reward is only valid for the model that
   produced it: the same node's gain correlates only +0.24 (Spearman) across two epochs,
   while `loss_before` correlates +0.59. The stable part of the target was the ignored part.
3. **The output was squashed and the loss chased outliers.** A sigmoid bounded the output
   against an unbounded signed target whose distribution is brutal -- skew +3.96, kurtosis
   +45.4, median gain +0.005 against a 99th percentile of +1.394. Under MSE a handful of
   extremes dominate every batch.

Naming, stated once: none of this is a DQN. There are no actions and no bootstrapping; the
original `train_step` regresses `Q(s) = R(s)` directly, which is plain supervised regression.
These classes are named for what they do. The `--dqn-model` flag and `dqn_model_type` keep
their names so existing configs and sweeps keep working.

The interface `DQNCapability` requires
------------------------------------
`device`, `parameters()`, `to()`, `batch_size`, `replay_buffer`, `train_step(transitions)`,
`predict_i_value(features, embedding)`, `save_checkpoint(path)`, `load_checkpoint(path)`.
This module additionally offers `observe()` / `learn()`, which the capability prefers when
present so that the *model* owns buffering -- see `Fix 2` below for why that placement
matters -- and declares `value_semantics` so the capability knows which way the output points.
"""

import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.capabilities.node_state import STATE_FEATURE_COUNT, STATE_LOSS_INDEX

#: Objectives. `rank` is the default because selection only ever consumes the *ordering* of
#: candidates, never the magnitude, and a pairwise ranking loss is invariant both to the
#: target's heavy tail and to any monotone rescaling of it.
GAIN_OBJECTIVES = ("rank", "huber")
DEFAULT_OBJECTIVE = "rank"

#: Fix 2. 16 batches rather than the old 10,000 samples (a full epoch of stale labels).
DEFAULT_BUFFER_SIZE = 512
#: Transitions older than this many observations are dropped unseen: the model that produced
#: them no longer exists, and the measured cross-epoch self-correlation of the target (+0.24)
#: says so quantitatively.
DEFAULT_MAX_TRANSITION_AGE = 2048
#: Sampling weight halves every this many observations of age.
DEFAULT_RECENCY_HALF_LIFE = 256

#: Pairs whose targets differ by less than this are treated as ties and skipped by `rank`.
#: The median gain is +0.005, so without this most pairs would contribute pure noise.
RANK_TIE_EPS = 1e-4

#: EMA rate for the running target standardisation used by `huber`.
TARGET_NORM_ALPHA = 0.01


def _stream(component, fallback_seed=0):
    """Seeded RNG for `component`, matching `DQNCapability._stream`.

    Sampling must not touch the process-global `random`, or RNG consumption anywhere
    upstream shifts which transitions get trained on.
    """
    from test_helpers.determinism import component_rng

    return component_rng(component, fallback_seed=fallback_seed)


def signed_log1p(values):
    """`sign(x) * log1p(|x|)`: compresses the tail without discarding the sign.

    The target spans roughly [-0.6, +1.4] at the 1st/99th percentiles with kurtosis +45, so
    a few samples otherwise dominate an MSE. Monotone, so it cannot reorder anything.
    """
    return torch.sign(values) * torch.log1p(values.abs())


def pairwise_ranking_loss(predictions, targets, tie_eps=RANK_TIE_EPS):
    """Logistic loss on every ordered pair whose targets actually differ.

    Returns `(loss, n_pairs)`. `n_pairs` is reported so a caller can notice the objective
    quietly training on almost nothing: with a median gain of +0.005 many pairs are ties, and
    a batch that yields none contributes no gradient at all.
    """
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    if predictions.numel() < 2:
        return predictions.sum() * 0.0, 0

    diff_pred = predictions.unsqueeze(0) - predictions.unsqueeze(1)
    diff_true = targets.unsqueeze(0) - targets.unsqueeze(1)

    # Upper triangle only: each unordered pair once.
    keep = torch.triu(torch.ones_like(diff_true, dtype=torch.bool), diagonal=1)
    keep = keep & (diff_true.abs() > tie_eps)
    n_pairs = int(keep.sum().item())
    if n_pairs == 0:
        return predictions.sum() * 0.0, 0

    sign = torch.sign(diff_true[keep])
    return F.softplus(-sign * diff_pred[keep]).mean(), n_pairs


class GainEstimatorBase(nn.Module):
    """Shared implementation of all three fixes.

    Fix 1 -- the cheap signal gets an unobstructed path to the output::

        value = state_linear(state_features) + tanh(residual_gate) * mlp(features, embedding)

    `state_linear` is initialised with a positive weight on the current-loss column
    (`node_state.STATE_LOSS_INDEX`) and `residual_gate` is initialised to zero, so an
    *untrained* model already ranks by current loss -- the +0.331 baseline -- and the learned
    residual can only add to it. That is the direct answer to a network that was handed the
    answer in its input and ignored it. The single gate subsumes the embedding-only gate the
    design sketched: gating the whole residual is what guarantees the baseline start, and
    `embedding_dim=0` or `use_embedding=False` removes the embedding pathway outright.

    Fix 3 -- no sigmoid. `forward` returns a raw scalar and `value_semantics` is
    ``"informativeness"``, so higher means more informative and `DQNCapability.get_i_value`
    passes it through unchanged rather than inverting it.
    """

    #: Higher output = more informative. The legacy family declares "legacy_inverted", where
    #: the output is `1 - sigmoid(Q)` and means the opposite of its own reward.
    value_semantics = "informativeness"

    def __init__(self, feature_dim, device, embedding_dim=512,
                 compressed_embedding_dim=64, state_dim=STATE_FEATURE_COUNT,
                 hidden_sizes=(128, 64), objective=DEFAULT_OBJECTIVE,
                 buffer_size=DEFAULT_BUFFER_SIZE,
                 max_transition_age=DEFAULT_MAX_TRANSITION_AGE,
                 recency_half_life=DEFAULT_RECENCY_HALF_LIFE,
                 batch_size=32, lr=1e-3, use_embedding=True, use_residual_mlp=True,
                 loss_index=STATE_LOSS_INDEX, loss_weight_init=1.0):
        super().__init__()
        if objective not in GAIN_OBJECTIVES:
            raise ValueError(
                f"unknown objective {objective!r}; choose from {', '.join(GAIN_OBJECTIVES)}"
            )
        self.device = device
        self.feature_dim = int(feature_dim)
        self.state_dim = max(0, min(int(state_dim), self.feature_dim))
        self.embedding_dim = int(embedding_dim) if use_embedding else 0
        self.compressed_embedding_dim = int(compressed_embedding_dim)
        self.objective = objective
        self.batch_size = int(batch_size)

        # --- Fix 1: the direct path ---------------------------------------------------
        if self.state_dim > 0:
            self.state_linear = nn.Linear(self.state_dim, 1)
            nn.init.zeros_(self.state_linear.weight)
            nn.init.zeros_(self.state_linear.bias)
            if 0 <= loss_index < self.state_dim:
                with torch.no_grad():
                    self.state_linear.weight[0, loss_index] = float(loss_weight_init)
        else:
            self.state_linear = None

        # --- the learned residual, gated shut at init ---------------------------------
        if use_residual_mlp:
            if self.embedding_dim > 0:
                self.embedding_processor = nn.Sequential(
                    nn.Linear(self.embedding_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.compressed_embedding_dim),
                    nn.ReLU(),
                )
                # The embedding is the only unbounded input group; `node_state.features` is
                # bounded to [0, 1] by construction and the static attributes are one-hot or
                # normalised quality scores.
                self.embedding_norm = nn.LayerNorm(self.compressed_embedding_dim)
                mlp_in = self.feature_dim + self.compressed_embedding_dim
            else:
                self.embedding_processor = None
                self.embedding_norm = None
                mlp_in = self.feature_dim

            layers = []
            width = mlp_in
            for size in hidden_sizes:
                layers += [nn.Linear(width, size), nn.ReLU()]
                width = size
            layers.append(nn.Linear(width, 1))
            self.mlp = nn.Sequential(*layers)
            self.residual_gate = nn.Parameter(torch.zeros(1))
        else:
            self.embedding_processor = None
            self.embedding_norm = None
            self.mlp = None
            self.residual_gate = None

        # --- Fix 2: the model owns its buffer -----------------------------------------
        self.replay_buffer = deque(maxlen=int(buffer_size))
        self.max_transition_age = int(max_transition_age)
        self.recency_half_life = max(1, int(recency_half_life))
        self._observations = 0

        # Running target standardisation for `huber`. Registered as buffers so they travel
        # with a checkpoint: a reloaded model that re-learns its target scale from scratch
        # would take a different path than one that never stopped.
        self.register_buffer("_target_mean", torch.zeros(()))
        self.register_buffer("_target_var", torch.ones(()))

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # ------------------------------------------------------------------ forward -------
    @property
    def wants_embeddings(self):
        """Whether `train_step` must stack the batch's embeddings.

        Inferring this from `embedding_processor is not None` was wrong for
        `LegacyGainAdapter`, which has no processor of its own because the *legacy* network
        consumes the embedding internally. The result was that every wrapped legacy model was
        handed `None` and trained on zeros -- silently for `basic`, whose `_process_embedding`
        substitutes zeros, and fatally for the other four, whose version dereferences
        `.shape` on it.
        """
        return self.embedding_processor is not None

    def _split(self, node_features):
        """`(all_features, state_block)`; the state block is the TRAILING slice.

        `DQNCapability._get_dqn_features` appends `node_state.features(...)` after the static
        attributes, so the trailing `state_dim` columns are the model-state block. Pinned by
        `tests/unit/test_gain_estimator.py`, because a reordering would silently point the
        loss-column initialisation at some unrelated attribute.
        """
        if node_features.dim() == 1:
            node_features = node_features.unsqueeze(0)
        state = node_features[..., -self.state_dim:] if self.state_dim > 0 else None
        return node_features, state

    def forward(self, node_features, node_embedding=None):
        features, state = self._split(node_features.to(self.device))

        value = torch.zeros(features.shape[0], 1, device=self.device)
        if self.state_linear is not None:
            value = value + self.state_linear(state)

        if self.mlp is not None:
            parts = [features]
            if self.embedding_processor is not None:
                if node_embedding is None:
                    compressed = torch.zeros(
                        features.shape[0], self.compressed_embedding_dim,
                        device=self.device,
                    )
                else:
                    embedding = node_embedding.to(self.device)
                    if embedding.dim() == 1:
                        embedding = embedding.unsqueeze(0)
                    compressed = self.embedding_norm(self.embedding_processor(embedding))
                parts.append(compressed)
            residual = self.mlp(torch.cat(parts, dim=1))
            value = value + torch.tanh(self.residual_gate) * residual

        return value

    def predict_value(self, node_features, node_embedding=None):
        """Raw informativeness score. Higher = expected to teach the model more."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            value = self(node_features, node_embedding)
        if was_training:
            self.train()
        return value

    def predict_i_value(self, node_features, node_embedding=None):
        """Kept for the interface `DQNCapability` already calls.

        No sigmoid and no inversion: `value_semantics` tells the capability that this number
        already points the right way.
        """
        return self.predict_value(node_features, node_embedding)

    # ------------------------------------------------------------------ buffering ----
    def observe(self, features, embedding, reward, step=None):
        """Record one (state, realised gain) observation.

        The capability calls this instead of appending to `replay_buffer` itself, which is
        what lets the *model* decide how old a label may be. Placement matters: the sampling
        used to live in the capability, so a model could shrink `maxlen` and still be handed
        a uniform draw across a whole epoch of stale rewards.
        """
        self._observations += 1
        self.replay_buffer.append((
            features.detach().to(self.device) if torch.is_tensor(features) else features,
            embedding.detach().to(self.device) if torch.is_tensor(embedding) else embedding,
            float(reward),
            self._observations if step is None else int(step),
        ))

    def sample_transitions(self, rng=None, count=None):
        """Recency-weighted draw, dropping anything past `max_transition_age`."""
        rng = rng or _stream("dqn.replay")
        count = count or self.batch_size
        now = self._observations

        fresh = [t for t in self.replay_buffer if now - t[3] <= self.max_transition_age]
        if len(fresh) < count:
            return []

        weights = [0.5 ** ((now - t[3]) / self.recency_half_life) for t in fresh]
        chosen, pool, pool_weights = [], list(fresh), list(weights)
        for _ in range(count):
            total = sum(pool_weights)
            if total <= 0:
                break
            threshold = rng.random() * total
            cumulative = 0.0
            for index, weight in enumerate(pool_weights):
                cumulative += weight
                if cumulative >= threshold:
                    chosen.append(pool.pop(index))
                    pool_weights.pop(index)
                    break
        return chosen

    def learn(self, rng=None):
        """Take one optimisation step if enough fresh transitions exist. Returns metrics."""
        transitions = self.sample_transitions(rng=rng)
        if not transitions:
            return None
        return self.train_step(transitions)

    # ------------------------------------------------------------------ training -----
    def _normalise_target(self, targets):
        """Standardise by an EMA of the target's mean and variance."""
        if self.training:
            with torch.no_grad():
                batch_mean = targets.mean()
                batch_var = targets.var(unbiased=False).clamp_min(1e-8)
                alpha = TARGET_NORM_ALPHA
                self._target_mean.mul_(1 - alpha).add_(alpha * batch_mean)
                self._target_var.mul_(1 - alpha).add_(alpha * batch_var)
        return (targets - self._target_mean) / self._target_var.clamp_min(1e-8).sqrt()

    def train_step(self, transitions):
        """One step on a batch of `(features, embedding, reward[, step])` tuples.

        Accepts 3-tuples as well as 4-tuples so the legacy capability path still works.
        Returns a dict rather than a bare float: `n_pairs` is the number the ranking
        objective actually trained on, and it is the one number that reveals the objective
        silently degenerating on near-tied targets.
        """
        if not transitions:
            return None

        features = torch.stack([
            t[0] if torch.is_tensor(t[0]) else torch.as_tensor(t[0]) for t in transitions
        ]).to(self.device)

        embeddings = None
        if self.wants_embeddings:
            width = self.embedding_dim
            embeddings = torch.stack([
                t[1] if torch.is_tensor(t[1])
                else torch.zeros(width, device=self.device)
                for t in transitions
            ]).to(self.device)

        targets = torch.tensor(
            [float(t[2]) for t in transitions], dtype=torch.float32, device=self.device
        )

        self.train()
        predictions = self(features, embeddings)

        n_pairs = 0
        if self.objective == "rank":
            loss, n_pairs = pairwise_ranking_loss(predictions, targets)
        else:
            loss = F.huber_loss(
                predictions.reshape(-1), self._normalise_target(signed_log1p(targets))
            )

        if not torch.is_tensor(loss) or not loss.requires_grad:
            return {"loss": 0.0, "n_pairs": n_pairs, "n_transitions": len(transitions)}

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "n_pairs": n_pairs,
            "n_transitions": len(transitions),
        }

    # ------------------------------------------------------------------ checkpoints --
    def save_checkpoint(self, filepath):
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "objective": self.objective,
            "observations": self._observations,
        }, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._observations = int(checkpoint.get("observations", 0))
        self.to(self.device)


class GainLinear(GainEstimatorBase):
    """The free baseline as a first-class model: a linear map on the state features only.

    Exists so "beat current loss" is a config change rather than an unwritten control. Every
    learned variant is judged against this, and it is also the sanity check on the harness --
    if this does not land near Spearman +0.33 on the gate, the measurement is wrong, not the
    model.
    """

    def __init__(self, feature_dim, device, **kwargs):
        kwargs.pop("use_residual_mlp", None)
        kwargs.pop("use_embedding", None)
        super().__init__(feature_dim, device, use_residual_mlp=False,
                         use_embedding=False, **kwargs)


class GainResidual(GainEstimatorBase):
    """The recommended default: linear baseline plus a gated MLP residual."""


class GainEnsemble(GainEstimatorBase):
    """`num_heads` residual heads over the shared trunk; predicts their mean.

    Tests whether variance reduction is what a target this noisy needs -- the same node's
    gain only self-correlates +0.24 across epochs. The spread across heads is a free
    disagreement signal, exposed by `predict_spread` for anyone who wants it later.
    """

    def __init__(self, feature_dim, device, num_heads=5, **kwargs):
        super().__init__(feature_dim, device, **kwargs)
        self.num_heads = int(num_heads)
        if self.mlp is None:
            raise ValueError("GainEnsemble needs the residual MLP; do not disable it")
        # Replace the single-output tail with `num_heads` independent tails.
        trunk = list(self.mlp)[:-1]
        final = list(self.mlp)[-1]
        self.mlp = nn.Sequential(*trunk).to(self.device)
        self.heads = nn.ModuleList([
            nn.Linear(final.in_features, 1) for _ in range(self.num_heads)
        ]).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def _head_outputs(self, node_features, node_embedding=None):
        features, state = self._split(node_features.to(self.device))
        base = torch.zeros(features.shape[0], 1, device=self.device)
        if self.state_linear is not None:
            base = base + self.state_linear(state)

        parts = [features]
        if self.embedding_processor is not None:
            if node_embedding is None:
                compressed = torch.zeros(
                    features.shape[0], self.compressed_embedding_dim, device=self.device
                )
            else:
                embedding = node_embedding.to(self.device)
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                compressed = self.embedding_norm(self.embedding_processor(embedding))
            parts.append(compressed)

        trunk = self.mlp(torch.cat(parts, dim=1))
        gate = torch.tanh(self.residual_gate)
        return base, torch.cat([gate * head(trunk) for head in self.heads], dim=1)

    def forward(self, node_features, node_embedding=None):
        base, heads = self._head_outputs(node_features, node_embedding)
        return base + heads.mean(dim=1, keepdim=True)

    def predict_spread(self, node_features, node_embedding=None):
        """Disagreement across heads. Zero at init, since every head is gated shut."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _base, heads = self._head_outputs(node_features, node_embedding)
        if was_training:
            self.train()
        return heads.std(dim=1, keepdim=True)


class LossEwmaRanker(GainEstimatorBase):
    """Rank candidates by the loss they last incurred. No learning at all.

    This is the control the whole estimator programme has to beat, made explicit. It reads
    one number that `NodeTrainingState` already stores -- a dictionary lookup, no forward
    pass, no buffer, no gradient -- and the 16-cell gate matrix found nothing that ranks
    realised learning gain meaningfully better: the best learned cell scored +0.177 against
    this signal's +0.171 in the same run.

    Implemented by freezing rather than by a separate code path, so it is provably the same
    thing the learned models start from: `GainLinear` at initialisation already outputs the
    current-loss column verbatim (see `GainEstimatorBase`), and here `train_step` is a no-op
    so it stays there. Any difference in a training run is therefore attributable to what the
    learned models *learn*, not to a different feature or a different scale.
    """

    def __init__(self, feature_dim, device, **kwargs):
        kwargs.pop("use_residual_mlp", None)
        kwargs.pop("use_embedding", None)
        kwargs.pop("objective", None)
        super().__init__(feature_dim, device, use_residual_mlp=False, use_embedding=False,
                         objective="huber", **kwargs)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def observe(self, features, embedding, reward, step=None):
        """Deliberately drops the observation: there is nothing to fit."""
        self._observations += 1

    def learn(self, rng=None):
        return None

    def train_step(self, transitions):
        """No-op. Reported rather than silent, so a caller cannot mistake it for training."""
        return {"loss": 0.0, "n_pairs": 0, "n_transitions": len(transitions or ())}


#: `--dqn-model` name -> class, for `create_dqn_model`.
GAIN_ESTIMATORS = {
    "loss_ewma": LossEwmaRanker,
    "gain_linear": GainLinear,
    "gain_residual": GainResidual,
    "gain_ensemble": GainEnsemble,
}

__all__ = [
    "LEGACY_ESTIMATORS", "LegacyGainAdapter", "build_estimator",
    "build_legacy_estimator",
    "DEFAULT_BUFFER_SIZE", "DEFAULT_MAX_TRANSITION_AGE", "DEFAULT_OBJECTIVE",
    "DEFAULT_RECENCY_HALF_LIFE", "GAIN_ESTIMATORS", "GAIN_OBJECTIVES", "GainEnsemble",
    "GainEstimatorBase", "GainLinear", "GainResidual", "LossEwmaRanker",
    "RANK_TIE_EPS",
    "pairwise_ranking_loss", "signed_log1p",
]


class LegacyGainAdapter(GainEstimatorBase):
    """Applies all three fixes to any of the five legacy models, without editing them.

    The plan called for a mixin threaded through `DQNModel` and the four classes in
    `EnhancedDQNModels`. This adapter is the same idea with the risk removed: instead of
    surgically retrofitting five bespoke architectures -- five more code paths to keep
    honest, five more chances to break the reproducibility of every past sweep -- it *reuses*
    the legacy network as the learned residual inside the fixed base::

        value = state_linear(state_features) + tanh(residual_gate) * legacy(features, embedding)

    So the legacy architecture is preserved exactly as the learned component, while the
    direct state path (Fix 1), the model-owned recency-weighted buffer (Fix 2) and the raw
    output with a rank-or-huber objective (Fix 3) all come from the base. The legacy files
    stay byte-identical, which is what keeps `--dqn-model basic` reproducible.

    That makes "is the fixed version better than the original?" a fair question: the two
    differ in the three fixes and nothing else, since the network in the middle is the same
    object either way.

    The legacy `forward` is used but its `predict_i_value`, `train_step`, `replay_buffer` and
    optimizer are all bypassed -- those are precisely where the three defects lived.
    """

    def __init__(self, legacy_model, feature_dim, device, **kwargs):
        kwargs.pop("use_residual_mlp", None)
        super().__init__(feature_dim, device, use_residual_mlp=False, **kwargs)
        self.legacy = legacy_model.to(device)
        self.residual_gate = nn.Parameter(torch.zeros(1, device=device))
        # Rebuilt so the legacy parameters and the new gate are actually optimised: the base
        # built its optimizer before either existed.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    @property
    def wants_embeddings(self):
        """True: the legacy network processes the embedding itself."""
        return self.embedding_dim > 0

    def forward(self, node_features, node_embedding=None):
        features, state = self._split(node_features.to(self.device))

        value = torch.zeros(features.shape[0], 1, device=self.device)
        if self.state_linear is not None:
            value = value + self.state_linear(state)

        # Materialise zeros rather than forwarding None. The five legacy models disagree
        # about a missing embedding -- `DQNModel` substitutes zeros, the other four
        # dereference `.shape` and raise -- and that difference is not this class's to
        # inherit.
        if node_embedding is None and self.embedding_dim > 0:
            node_embedding = torch.zeros(
                features.shape[0], self.embedding_dim, device=self.device
            )
        elif node_embedding is not None:
            node_embedding = node_embedding.to(self.device)
            if node_embedding.dim() == 1:
                node_embedding = node_embedding.unsqueeze(0)

        residual = self.legacy(features, node_embedding)
        if not torch.is_tensor(residual):
            residual = getattr(residual, "logits", residual)
        residual = residual.reshape(features.shape[0], -1)[:, :1]
        return value + torch.tanh(self.residual_gate) * residual


#: The five original architectures, by `--dqn-model` name.
LEGACY_ESTIMATORS = ("basic", "residual", "attention", "conv_embedding", "ensemble")


def build_legacy_estimator(model_type, feature_dim, device, embedding_dim=512, **kwargs):
    """Construct one of the five originals, exactly as before."""
    if model_type == "basic":
        from models.DQNModel import DQNModel
        return DQNModel(feature_dim, device, embedding_dim=embedding_dim)
    if model_type == "residual":
        from models.EnhancedDQNModels import ResidualDQNModel
        return ResidualDQNModel(feature_dim, device, embedding_dim=embedding_dim, **kwargs)
    if model_type == "attention":
        from models.EnhancedDQNModels import AttentionDQNModel
        return AttentionDQNModel(feature_dim, device, embedding_dim=embedding_dim, **kwargs)
    if model_type == "conv_embedding":
        from models.EnhancedDQNModels import ConvEmbeddingDQN
        return ConvEmbeddingDQN(feature_dim, device, embedding_dim=embedding_dim, **kwargs)
    if model_type == "ensemble":
        from models.EnhancedDQNModels import EnsembleDQNModel
        return EnsembleDQNModel(feature_dim, device, embedding_dim=embedding_dim, **kwargs)
    raise ValueError(f"Unsupported DQN model type: {model_type}")


def build_estimator(model_type, feature_dim, device, embedding_dim=512,
                    apply_fixes=False, **kwargs):
    """The single I-value estimator factory.

    This dispatch existed in three places -- `test_hierarchical.create_dqn_model`,
    `DQNCapability._initialize_dqns`, and partly in the web UI's config validation -- which is
    how `gain_*` could be selectable from the CLI and still never reach the capability. One
    factory, three callers.

    `apply_fixes` wraps a legacy architecture in `LegacyGainAdapter`. Without it a legacy name
    returns the original object and prints a deprecation notice, because that configuration is
    measured to rank learning gain at Spearman +0.01 against +0.33 for current loss.
    """
    gain_keys = ("state_dim", "objective", "buffer_size", "max_transition_age",
                 "recency_half_life", "loss_weight_init", "num_heads")
    gain_kwargs = {key: kwargs.pop(key) for key in gain_keys if key in kwargs}

    if model_type in GAIN_ESTIMATORS:
        return GAIN_ESTIMATORS[model_type](
            feature_dim, device, embedding_dim=embedding_dim, **gain_kwargs, **kwargs
        )

    legacy = build_legacy_estimator(
        model_type, feature_dim, device, embedding_dim=embedding_dim, **kwargs
    )
    if apply_fixes:
        gain_kwargs.pop("num_heads", None)
        return LegacyGainAdapter(
            legacy, feature_dim, device, embedding_dim=embedding_dim, **gain_kwargs
        )
    print(
        f"[deprecated] --dqn-model {model_type} without --dqn-fixes keeps the original "
        f"1-sigmoid(Q) output, 10,000-sample replay buffer and MSE objective. Measured, that "
        f"combination ranks realised learning gain at Spearman +0.01 where a sample's current "
        f"loss ranks it at +0.33. Pass --dqn-fixes, or --dqn-model gain_residual."
    )
    return legacy
