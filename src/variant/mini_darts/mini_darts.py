from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Node(nn.Module):
    def __init__(self):
        super().__init__()
        self.edges = nn.ModuleList()


class Edge(nn.Module):
    def __init__(self, available_operations: List[str], out_channels: int = 16):
        super().__init__()
        self.mixed_operation = MixedOperation(
            available_operations=available_operations, out_channels=out_channels
        )
        self.alphas = nn.Parameter(
            torch.randn(len(available_operations)) * 0.001
        )  # 0.001 to initialize alphas to a small values

    def forward(self, x):
        return self.mixed_operation(x, F.softmax(self.alphas, dim=-1))


class MixedOperation(nn.Module):
    def __init__(self, available_operations: List[str], out_channels: int = 16):
        super().__init__()
        self.ops = nn.ModuleList()

        for operation_name in available_operations:
            if operation_name == "max_pool_3x3":
                op = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            elif operation_name == "avg_pool_3x3":
                op = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            elif operation_name == "conv_3x3":
                op = nn.Sequential(
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        groups=out_channels,
                    ),
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU(),
                )
            elif operation_name == "identity":
                op = nn.Identity()
            else:
                raise ValueError(f"Operation {operation_name} not supported")

            self.ops.append(op)

    def forward(self, x, alphas):
        return sum(op(x) * alpha for op, alpha in zip(self.ops, alphas))


class Cell(nn.Module):
    def __init__(
        self,
        available_operations: List[str],
        num_intermediate_nodes: int = 6,
        num_input_nodes: int = 2,
        num_output_nodes: int = 1,
    ):
        super().__init__()
        self.available_operations = available_operations

        self.num_input_nodes = num_input_nodes
        self.num_intermediate_nodes = max(
            num_intermediate_nodes - self.num_input_nodes, 1
        )  # 1 to ensure there is at least one intermediate node
        self.num_output_nodes = num_output_nodes
        self.num_nodes = (
            self.num_intermediate_nodes + self.num_input_nodes + self.num_output_nodes
        )

        self.nodes = nn.ModuleList(Node() for _ in range(self.num_nodes))

        self._initialize_edges()

    def _initialize_edges(self):
        for i in range(self.num_input_nodes, self.num_nodes):
            for _ in range(i):
                edge = Edge(available_operations=self.available_operations)
                self.nodes[i].edges.append(edge)

    def forward(self, input_features):
        node_outputs = []

        for _ in range(self.num_input_nodes):
            node_outputs.append(input_features)

        for i in range(
            self.num_input_nodes, self.num_intermediate_nodes - self.num_output_nodes
        ):
            node_inputs = []
            for j, edge in enumerate(self.nodes[i].edges):  # edges to node i
                node_inputs.append(edge.mixed_operation(node_outputs[j]))

            node_outputs.append(sum(node_inputs))

        return node_outputs[-self.num_output_nodes :]


class DARTS(pl.LightningModule):
    def __init__(self, available_operations: List[str], num_nodes: int = 6):
        super().__init__()
        self.available_operations = available_operations
        self.cell = Cell(
            num_intermediate_nodes=num_nodes, available_operations=available_operations
        )


def mini_darts_example():
    available_operations = [
        "max_pool_3x3",
        "conv_3x3",
        "avg_pool_3x3",
        "identity",
    ]
    model = DARTS(available_operations=available_operations)
    print(model)


if __name__ == "__main__":
    mini_darts_example()
