# The I-value gate: the DQN does not predict learning gain

Run `run_outputs/ivalue_gate` (2026-08-21): 40,000-node pool, 2 epochs, exactly 10,000 nodes
per epoch, `--ivalue-reward learning_gain --ivalue-state-features --ivalue-diagnostic`.
20,000 paired observations of (I-value the traversal saw when it chose a node, per-sample
loss reduction that training on it actually produced).

**Verdict: FAIL.** Reported so the 8-arm design sweep is not run on an estimator that has
nothing to exploit.

| quantity | value |
|---|---|
| spearman(predicted I-value, realised gain) | **+0.0102** |
| AUROC(predicted ranks top-decile gain) | 0.531 (chance) |
| spearman(**current loss**, realised gain) | **+0.3306** |
| R^2 of a straight line, current loss -> gain | 0.398 |
| per-epoch spearman | epoch 0: **-0.0159**, epoch 1: +0.0387 |

## The finding is not "learning gain is unpredictable"

It is the opposite, and that is what makes it actionable. The signal is strong and nearly
free: **a sample's current loss predicts its realised learning gain at Spearman +0.33
(R^2 = 0.40)**, and the training loop already has that number. Learning gain is bounded by
how much loss there is to remove, so this is close to a tautology -- which is exactly why it
is a hard baseline.

The DQN reaches +0.010 against that +0.33. Worse, **current loss is one of its input
features** and its output is essentially uncorrelated with it: `corr(predicted, loss_before)
= -0.009`. The answer was handed to the network and it did not use it.

The output is not collapsed either -- it spans [0.0000, 0.9998] with sd 0.071 around a mean
of 0.525, and 19,844 of 20,000 values are distinct. It is producing varied, confident,
uninformative predictions. It is fitting noise.

## Three mechanical causes, in order of size

**1. The informative features are swamped.** `DQNModel` compresses the 512-d face embedding
to 64 dims and concatenates it with the `feature_dim` vector, so the input to `fc1` is
95 dims of which the 6 model-state features are **6.3%** and the embedding pathway is 67%.
The embedding is a static property of the image and carries no information about what the
detector currently knows, so two thirds of the input is irrelevant to the target by
construction.

**2. The replay buffer serves stale labels for a non-stationary target.**
`replay_buffer = deque(maxlen=10000)` holds up to a full epoch of rewards, and a reward is
only valid for the model that produced it. Measured directly: for the 362 nodes trained on
in both epochs, the same node's gain correlates only **+0.257** between epoch 0 and epoch 1.
The network is regressing labels that were true of a model that no longer exists.

**3. The output is squashed.** `predict_i_value` applies a sigmoid, so Q is pushed toward a
bounded range while the target (gain, sd 0.325, both signs) is not. Under MSE this pulls the
fit toward the mean.

## What to do instead

1. **Select by current loss (or a cheap expected-gain proxy) and drop the DQN from the
   selection path.** Free, already computed, +0.33 against the DQN's +0.01. Any future
   learned estimator must beat *this*, not random -- and that is the comparison to publish.
   Note the honest framing: this is loss-based hard-example mining, which is not novel, so
   the contribution would have to be the graph/traversal machinery around it.
2. If the DQN is kept, fix the three causes above before re-gating: rebalance or remove the
   embedding pathway, shrink the replay buffer hard (or drop it -- a single on-policy
   regression step per batch matches a non-stationary target better than a 10,000-sample
   buffer), and remove the sigmoid so the head can span the target's range.
3. Re-run this gate after any such change. It costs ~20 minutes on the 40k pool and it is
   the cheapest experiment in this repository.

## Note on a misleading smoke result

A 300-node, 100-sample smoke run of the same code reported spearman **+0.456** and beat the
current-loss baseline. It did not replicate at 20,000 rows on the real pool. Do not read a
gate result off a toy run; the whole point of the gate is that it is cheap enough to run at
a size where the answer means something.
