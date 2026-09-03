from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class PredictionBundle:
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    uncertainty: Dict[str, torch.Tensor] = field(default_factory=dict)
    evidence: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    member_logits: Optional[torch.Tensor] = None
    gp_variance: Optional[torch.Tensor] = None

    def with_predictions(self) -> "PredictionBundle":
        if self.predictions is None:
            self.predictions = (self.probabilities > 0.5).float()
        return self

