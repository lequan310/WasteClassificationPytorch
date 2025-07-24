# import torch
# from torchmetrics.classification import MulticlassAccuracy

# from base import BaseMetric


# class Accuracy(BaseMetric):
#     """
#     Calculates multiclass accuracy.
#     """

#     def __init__(
#         self, num_classes: int, device: str = "auto", average="macro", name="accuracy"
#     ):
#         super().__init__(name=name)

#         if device == "auto":
#             device = "cuda" if torch.cuda.is_available() else "cpu"

#         self.metric = MulticlassAccuracy(num_classes=num_classes, average=average).to(
#             device
#         )

#     def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
#         """
#         Args:
#             output (torch.Tensor): Model output of shape (batch_size, num_classes).
#             target (torch.Tensor): Ground truth of shape (batch_size,).
#         Returns:
#             float: The accuracy score for the batch.
#         """
#         pred = torch.argmax(output, dim=-1)
#         return self.metric(pred, target)

import torch
from torchmetrics.classification import MulticlassAccuracy

from base import BaseMetric


class Accuracy(BaseMetric):
    """
    Calculates the Accuracy score using torchmetrics correctly.
    """

    def __init__(
        self, num_classes: int, device: str = "auto", average="macro", name="accuracy"
    ):
        super().__init__(device=device, name=name)

        self.metric = MulticlassAccuracy(num_classes=num_classes, average=average).to(
            self.device
        )

    def update(self, output: torch.Tensor, target: torch.Tensor):
        """
        Updates the state of the metric with new predictions and targets.
        Args:
            output (torch.Tensor): Model output of shape (batch_size, num_classes).
            target (torch.Tensor): Ground truth of shape (batch_size,).
        """
        # Ensure tensors are on the same device as the metric
        pred = torch.argmax(output, dim=-1)
        self.metric.update(pred, target)  # Just update, don't return anything

    def compute(self):
        """
        Computes the Accuracy score over all batches seen since the last reset.
        Returns:
            float: The final Accuracy score.
        """
        # 2. Compute the final result from all accumulated data.
        return self.metric.compute().item()

    def reset(self):
        """
        Resets the internal state of the metric.
        """
        # 3. Reset for the next epoch.
        self.metric.reset()
