import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicSizeConv2d(nn.Module):
    def __init__(self, kernel_size, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def to_trainable(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        weight: torch.Tensor = torch.randn(
            x.size(1), x.size(1), self.kernel_size, self.kernel_size, device=x.device
        )
        return F.conv2d(x, weight, padding=self.padding)


class DynamicSizeSeparableConv2d(nn.Module):
    def __init__(self, kernel_size, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def to_trainable(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Depthwise convolution
        depthwise_weight = torch.randn(
            x.size(1), 1, self.kernel_size, self.kernel_size, device=x.device
        )

        depthwise_output = F.conv2d(
            x, depthwise_weight, padding=self.padding, groups=x.size(1)
        )

        # Pointwise convolution
        pointwise_weight = torch.randn(x.size(1), x.size(1), 1, 1, device=x.device)

        return F.conv2d(depthwise_output, pointwise_weight, padding=0)


class DynamicSizeDilatedConv2d(nn.Module):
    def __init__(self, kernel_size, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation

    def to_trainable(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                dilation=self.dilation,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        weight = torch.randn(
            x.size(1), x.size(1), self.kernel_size, self.kernel_size, device=x.device
        )

        return F.conv2d(x, weight, padding=self.padding, dilation=self.dilation)


class MaxPool(nn.MaxPool2d):
    def to_trainable(self, in_channels):
        return self


class AvgPool(nn.AvgPool2d):
    def to_trainable(self, in_channels):
        return self


class Identity(nn.Identity):
    def to_trainable(self, in_channels):
        return self


class ZeroOp(nn.Module):
    def to_trainable(self, in_channels):
        return self  # Zero operation remains the same

    def forward(self, x):
        return torch.zeros_like(x)
