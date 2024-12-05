from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_alphas(num_available_operations):
    return nn.Parameter(
        torch.randn(num_available_operations) * 0.001
    )  # 0.001 to initialize alphas to a small values


def default_stem(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class Node(nn.Module):
    def __init__(self):
        super().__init__()
        self.edges = nn.ModuleList()


class Edge(nn.Module):
    def __init__(
        self,
        available_operations: List[str],
        in_channels: int,
        out_channels: int = 16,
        alphas=None,
    ):
        super().__init__()
        self.mixed_operation = MixedOperation(
            available_operations=available_operations,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.alphas = alphas or default_alphas(len(available_operations))

    def forward(self, x):
        weights = F.softmax(self.alphas, dim=-1)
        return self.mixed_operation(x, weights)


class MixedOperation(nn.Module):
    def __init__(
        self, available_operations: List[str], in_channels: int, out_channels: int = 16
    ):
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
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU(),
                )
            elif operation_name == "sep_conv_3x3":
                op = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        padding=1,
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU(),
                )
            elif operation_name == "identity":
                op = nn.Identity() if in_channels == out_channels else None
            elif operation_name == "none":
                op = NoConnectionOp()
            else:
                raise ValueError(f"Operation {operation_name} not supported")

            if op is not None:
                self.ops.append(op)

    def forward(self, x, alphas):
        return sum(op(x) * alpha for op, alpha in zip(self.ops, alphas))


class Cell(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        available_operations: List[str],
        num_input_nodes: int = 2,
        num_intermediate_nodes: int = 4,
        num_output_nodes: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.available_operations = available_operations

        self.num_input_nodes = num_input_nodes
        self.num_intermediate_nodes = num_intermediate_nodes
        self.num_output_nodes = num_output_nodes
        self.num_nodes = (
            self.num_input_nodes + self.num_intermediate_nodes + self.num_output_nodes
        )

        self.stem = default_stem(in_channels, out_channels)

        self.nodes = nn.ModuleList(Node() for _ in range(self.num_nodes))

        self._initialize_edges()

    def _initialize_edges(self):
        for node_index in range(self.num_input_nodes, self.num_nodes):
            for edge_index in range(node_index):
                edge = Edge(available_operations=self.available_operations)
                self.nodes[node_index].edges.append(edge)

    def forward(self, input_features):
        every_node_output = []

        for _ in range(self.num_input_nodes):
            every_node_output.append(input_features)

        for node_index in range(
            self.num_input_nodes,
            self.num_input_nodes + self.num_intermediate_nodes,
        ):
            current_node_inputs = []
            for edge_index, edge in enumerate(self.nodes[node_index].edges):
                current_node_inputs.append(
                    edge.mixed_operation(every_node_output[edge_index])
                )

            every_node_output.append(sum(current_node_inputs))
            # I wonder if other operation than sum() would give better search results?

        return every_node_output[-self.num_output_nodes :]


class NoConnectionOp(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class DARTS(pl.LightningModule):
    def __init__(self, available_operations: List[str], num_nodes: int = 6):
        super().__init__()
        self.available_operations = available_operations
        self.cell = Cell(
            num_intermediate_nodes=num_nodes,
            available_operations=available_operations,
        )

    def get_weights(self):
        return list(self.cell.nodes.parameters())

    def get_alphas(self):
        return list(self.cell.nodes[0].edges[0].alphas.parameters())


class DARTSTrainer(pl.LightningModule):
    def __init__(self, model: DARTS, config: dict):
        super().__init__()
        self.model = model
        self.config = config


def example():
    available_operations = [
        "none",
        "max_pool_3x3",
        "conv_3x3",
        "avg_pool_3x3",
        "identity",
    ]
    model = DARTS(available_operations=available_operations)
    print(model)
    trainer = DARTSTrainer(model=model, config={})
    print(trainer)


if __name__ == "__main__":
    example()
