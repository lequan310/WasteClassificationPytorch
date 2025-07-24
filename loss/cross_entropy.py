import torch
from torch import nn


class CrossEntropy(nn.Module):
    def __init__(self, **kwargs):
        """Initializes the CrossEntropy loss function."""

        super().__init__()
        self.loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        return self.loss(logits, labels)
