# from abc import abstractmethod


# class BaseMetric:
#     """
#     Base class for all metrics
#     """

#     def __init__(self, name=None, *args, **kwargs):
#         """
#         Args:
#             name (str | None): metric name to use in logger and writer.
#         """
#         self.name = name if name is not None else type(self).__name__

#     @abstractmethod
#     def __call__(self, **batch):
#         """
#         Defines metric calculation logic for a given batch.
#         Can use external functions (like TorchMetrics) or custom ones.
#         """
#         raise NotImplementedError()

from abc import abstractmethod

import torch


class BaseMetric:
    """
    Base class for all metrics.
    """

    def __init__(self, device, name=None):
        self.name = name if name is not None else self.__class__.__name__

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    @abstractmethod
    def update(self, output: torch.Tensor, target: torch.Tensor):
        """Update the metric's state."""
        raise NotImplementedError

    @abstractmethod
    def compute(self):
        """Compute the final metric value."""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset the metric's state."""
        raise NotImplementedError
