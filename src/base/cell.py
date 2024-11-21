from typing import List

import torch
import torch.nn as nn


class Cell(nn.Module):
    def __init__(self, derived_architecture: list, in_channels: int, stride: int):
        super().__init__()
        self.derived_architecture = derived_architecture
        self.stride = stride

        # Preprocess inputs to have the same channels and resolution
        self.preprocess0 = PreprocessX(in_channels, stride=1 if stride == 1 else 2)
        self.preprocess1 = PreprocessX(in_channels, stride=1)

        # Build the cell's internal nodes
        self.nodes = nn.ModuleList()
        for node_ops in derived_architecture:
            node = Node(operations=node_ops, in_channels=in_channels, stride=stride)
            self.nodes.append(node)

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        # Preprocess both input states
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]

        # Process each node
        for node in self.nodes:
            # Each node takes all previous states as input
            out = node(states)
            states.append(out)

        # Concatenate all intermediate nodes (excluding input states)
        return torch.cat(states[2:], dim=1)


class PreprocessX(nn.Module):
    """Preprocess input states to have matching dimensions"""

    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return self.op(x)


class Node(nn.Module):
    """A node in the cell that combines multiple inputs"""

    def __init__(self, operations: list, in_channels: int, stride: int):
        super().__init__()
        self.ops = nn.ModuleList()
        for input_idx, op_class in operations:
            op = op_class.to_trainable(in_channels)
            if stride > 1 and input_idx in [0, 1]:  # Apply stride only to input nodes
                op = nn.Sequential(op, nn.MaxPool2d(2, stride=2))
            self.ops.append((input_idx, op))  # type: ignore

    def forward(self, states: List[torch.Tensor]) -> torch.Tensor:
        # Sum all input operations
        out = torch.sum(op(states[input_idx]) for input_idx, op in self.ops)  # type: ignore
        return out
