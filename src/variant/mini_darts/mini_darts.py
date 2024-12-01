from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Node(nn.Module):
    def __init__(self):
        super().__init__()
        self.edges = nn.ModuleList()


class Edge(nn.Module):
    def __init__(self, available_operations: List[str]):
        super().__init__()
        self.mixed_operation = MixedOperation(available_operations=available_operations)
        self.alphas = nn.Parameter(
            torch.randn(len(available_operations)) * 0.001
        )  # 0.001 to initialize alphas to a small values


class MixedOperation(nn.Module):
    def __init__(self, available_operations: List[str]):
        super().__init__()
        self.ops = available_operations


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
        for i in range(self.num_intermediate_nodes):
            for _ in range(i):
                edge = Edge(available_operations=self.available_operations)
                self.nodes[i].edges.append(edge)

    def forward(self, input_features):
        node_outputs = []

        for _ in range(self.num_input_nodes):
            node_outputs.append(input_features)

        for i in range(self.num_intermediate_nodes):
            node_inputs = []
            for j, edge in enumerate(self.nodes[i].edges):  # edges to node i
                node_inputs.append(
                    edge.mixed_operation(node_outputs[j])
                )  # TODO: won't work yet

            node_outputs.append(sum(node_inputs))

        pass


class DARTS(nn.Module):
    def __init__(self, available_operations: List[str], num_nodes: int = 6):
        super().__init__()
        self.available_operations = available_operations
        self.cell = Cell(
            num_intermediate_nodes=num_nodes, available_operations=available_operations
        )


def mini_darts_example():
    available_operations = [
        nn.MaxPool2d,
        nn.Conv2d,
        nn.AvgPool2d,
        nn.Identity,
    ]
    model = DARTS(available_operations=available_operations)
    print(model)


if __name__ == "__main__":
    mini_darts_example()
