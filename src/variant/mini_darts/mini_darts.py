from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def reduction_stride(reduction: bool):
    return 2 if reduction else 1


def default_alphas(num_available_operations):
    return nn.Parameter(
        torch.randn(num_available_operations) * 0.001
    )  # 0.001 to initialize alphas to a small values


def default_stem(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
        in_channels: int = 16,
        out_channels: int = 16,
        alphas=None,
        reduction: bool = False,
    ):
        super().__init__()
        self.mixed_operation = MixedOperation(
            available_operations=available_operations,
            in_channels=in_channels,
            out_channels=out_channels,
            reduction=reduction,
        )
        self.alphas = alphas or default_alphas(len(available_operations))

    def forward(self, x):
        weights = F.softmax(self.alphas, dim=-1)
        return self.mixed_operation(x, weights)


class MixedOperation(nn.Module):
    def __init__(
        self,
        available_operations: List[str],
        in_channels: int,
        out_channels: int = 16,
        reduction: bool = False,
    ):
        super().__init__()
        self.ops = nn.ModuleList()
        self.reduction = reduction
        stride = reduction_stride(reduction)

        for operation_name in available_operations:
            if operation_name == "max_pool_3x3":
                op = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
            elif operation_name == "avg_pool_3x3":
                op = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
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
        in_channels_prev: int,
        in_channels_prev_prev: int,
        out_channels: int,
        available_operations: List[str],
        num_input_nodes: int = 2,
        num_intermediate_nodes: int = 4,
        num_output_nodes: int = 1,
        reduction: bool = False,
    ):
        super().__init__()

        self.available_operations = available_operations

        self.num_input_nodes = num_input_nodes
        self.num_intermediate_nodes = num_intermediate_nodes
        self.num_output_nodes = num_output_nodes
        self.num_nodes = (
            self.num_input_nodes + self.num_intermediate_nodes + self.num_output_nodes
        )

        self.preprocess_prev_prev = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels_prev_prev,
                out_channels=out_channels,
                kernel_size=1,
                stride=reduction_stride(reduction),
                padding=0,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

        self.preprocess_prev = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels_prev,
                out_channels=out_channels,
                kernel_size=1,
                stride=reduction_stride(reduction),
                padding=0,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

        self.nodes = nn.ModuleList(Node() for _ in range(self.num_nodes))

        self._initialize_edges()

    def _initialize_edges(self):
        for node_index in range(self.num_input_nodes, self.num_nodes):
            for edge_index in range(node_index):
                edge = Edge(
                    available_operations=self.available_operations,
                    in_channels=self.channels,
                    out_channels=self.channels,
                )
                self.nodes[node_index].edges.append(edge)

    def forward(self, input_features):
        s0 = self.preprocess_prev_prev(input_features)
        s1 = self.preprocess_prev(input_features)

        every_node_output = [s0, s1]

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
    def __init__(
        self,
        available_operations: List[str],
        layers: int = 20,
        in_channels: int = 3,
        init_channels: int = 16,
        reduction_cell_indices=None,
    ):
        super().__init__()
        self.available_operations = available_operations
        self.stem = default_stem(in_channels, init_channels)

        self.reduction_cell_indices = reduction_cell_indices or [
            layers // 3,
            2 * layers // 3,
        ]

        self.cells = nn.ModuleList()

        current_channels = init_channels
        prev_prev_channels = init_channels
        prev_channels = init_channels

        for layer_index in range(layers):
            is_reduction_cell = layer_index in self.reduction_cell_indices

            if is_reduction_cell:
                prev_prev_channels = prev_channels
                prev_channels = current_channels * 2
                current_channels *= 2
            else:
                prev_prev_channels = prev_channels
                prev_channels = current_channels

            self.cells.append(
                Cell(
                    in_channels_prev=prev_channels,
                    in_channels_prev_prev=prev_prev_channels,
                    out_channels=current_channels,
                    available_operations=available_operations,
                    reduction=is_reduction_cell,
                )
            )

    def forward(self, x):
        x = self.stem(x)

        for cell in self.cells:
            x = cell(x)

        return x


class DARTSTrainer(pl.LightningModule):
    def __init__(self, model: DARTS, config: dict):
        super().__init__()
        self.model = model
        self.config = config
        self.weights_optimizer = Adam(
            self.model.parameters(), lr=config["training"]["lr"]
        )

    def training_step(self, batch, batch_idx):
        """
        bi-level optimization
        """

        # weights
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss


def example():
    available_operations = [
        "none",
        "max_pool_3x3",
        "conv_3x3",
        "sep_conv_3x3",
        "avg_pool_3x3",
        "identity",
    ]
    model = DARTS(available_operations=available_operations)
    print(model)
    trainer = DARTSTrainer(model=model, config={})
    print(trainer)


if __name__ == "__main__":
    example()
