"""Training-time uncertainty logging, kept off the gradient path.

MC dropout used to run *inside* the training loss path, but only every
``uncertainty_train_frequency`` batches. That made the optimization objective
change periodically -- on those batches the loss was computed from an average of N
stochastic passes, on every other batch from a single deterministic one. A
non-stationary objective is not something you want silently switched on and off
mid-epoch.

So the loss now always uses one deterministic pass, and MC statistics are gathered
here under ``torch.no_grad()`` for logging only. That also makes the training step
cheaper, since the N extra passes no longer build a graph.

These scalars are for monitoring. Per-sample uncertainty for the benchmark comes
from the evaluation path, not from batch means.
"""

import torch


def uncertainty_summary_for_logging(model, prediction_bundle, inputs, nodes):
    """Batch-mean uncertainty scalars for this step.

    Adds MC-dropout statistics when the model is configured for them and the model
    actually has stochastic dropout to sample -- ``mc_dropout_available()`` guards
    against the case where every dropout module has ``p=0``, which would report
    identically zero variance as though it were a measurement.
    """
    summary = dict(model.summarize_uncertainty(prediction_bundle))

    samples = getattr(model, 'mc_dropout_samples', 0)
    if samples > 1 and getattr(model, 'mc_dropout_available', lambda: False)():
        with torch.no_grad():
            mc_bundle = model.forward_with_uncertainty(
                inputs,
                nodes=nodes,
                update_precision=False,  # never accumulate SNGP precision from MC passes
                use_mc_dropout=True,
            )
        summary.update(model.summarize_uncertainty(mc_bundle))

    return summary
