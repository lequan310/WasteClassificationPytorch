from abc import abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models.
    """

    @abstractmethod
    def forward(self, *inputs: Any) -> torch.Tensor:
        """
        Forward pass logic.

        Args:
            *inputs: Variable length argument list of input tensors

        Returns:
            Model output tensor

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        Model prints with number of trainable parameters.

        Returns:
            String representation of the model with parameter count
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum(np.prod(p.size()) for p in model_parameters)
        return f"{super().__str__()}\nTrainable parameters: {params:,}"
