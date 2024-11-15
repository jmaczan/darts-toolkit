import torch
import torch.nn as nn


class AuxiliaryHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Ensure input is properly shaped before pooling
            nn.ReLU(inplace=True),
            # Adjust pooling parameters if needed
            nn.AvgPool2d(
                kernel_size=5, stride=2, padding=2
            ),  # Modified stride and padding
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add shape validation
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")

        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
