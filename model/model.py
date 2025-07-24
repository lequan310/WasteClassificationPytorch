from math import ceil

import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation

from base import BaseModel


class MBConv(BaseModel):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) block.
    This is the main building block of EfficientNet.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the depthwise convolution kernel
        stride: Stride for the depthwise convolution
        expand_ratio: Expansion ratio for the pointwise convolution
        se_ratio: Squeeze-and-excitation ratio
        drop_connect_rate: Drop connect rate for stochastic depth
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand_ratio = expand_ratio
        reduced_dim = max(1, int(in_channels * se_ratio))

        # Stochastic Depth (Drop Connect)
        self.drop_connect_rate = drop_connect_rate

        layers = []
        # Expansion phase (Pointwise convolution)
        if self.expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(),
                ]
            )

        # Depthwise convolution
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # Squeeze-and-Excitation block
                SqueezeExcitation(hidden_dim, reduced_dim),
                # Projection phase (Pointwise convolution)
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def _stochastic_depth(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth (drop connect) regularization."""
        if not self.training or self.drop_connect_rate == 0:
            return x

        batch_size = x.shape[0]
        keep_prob = 1 - self.drop_connect_rate
        random_tensor = keep_prob + torch.rand(
            [batch_size, 1, 1, 1], dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self._stochastic_depth(self.conv(x))
        else:
            return self.conv(x)


class EfficientNet(BaseModel):
    """
    EfficientNet-B0 model architecture.

    Args:
        num_classes: Number of output classes
        width_mult: Width multiplier for scaling channels
        depth_mult: Depth multiplier for scaling layers
        dropout_rate: Dropout rate for the classifier
    """

    def __init__(
        self,
        num_classes: int = 9,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        # (expand_ratio, channels, repeats, stride, kernel_size)
        settings = [
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]

        def _round_channels(channels: int, multiplier: float) -> int:
            """Round number of channels based on width multiplier."""
            if multiplier == 1.0:
                return int(channels)
            divisor = 8
            channels *= multiplier
            new_channels = max(
                divisor, int(channels + divisor / 2) // divisor * divisor
            )
            if new_channels < 0.9 * channels:
                new_channels += divisor
            return int(new_channels)

        def _round_repeats(repeats: int, multiplier: float) -> int:
            """Round number of layer repeats based on depth multiplier."""
            if multiplier == 1.0:
                return int(repeats)
            return int(ceil(multiplier * repeats))

        in_channels = 32
        last_channels = 1280
        in_channels = _round_channels(in_channels, width_mult)
        self.last_channels = _round_channels(last_channels, width_mult)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )

        # Build MBConv blocks
        blocks = []
        total_blocks = sum(_round_repeats(s[2], depth_mult) for s in settings)
        block_idx = 0

        for expand_ratio, C, R, S, K in settings:
            out_channels = _round_channels(C, width_mult)
            repeats = _round_repeats(R, depth_mult)
            for r in range(repeats):
                stride = S if r == 0 else 1
                drop_connect_rate = dropout_rate * float(block_idx) / total_blocks
                blocks.append(
                    MBConv(
                        in_channels,
                        out_channels,
                        kernel_size=K,
                        stride=stride,
                        expand_ratio=expand_ratio,
                        drop_connect_rate=drop_connect_rate,
                    )
                )
                in_channels = out_channels
                block_idx += 1
        self.blocks = nn.Sequential(*blocks)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, self.last_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.last_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.last_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
