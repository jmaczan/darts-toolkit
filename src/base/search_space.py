from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn


class BaseSearchSpace(nn.Module, ABC):
    """
    Abstract base class for all DARTS search spaces.
    Provides the basic interface that all DARTS variants should implement.
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        candidate_operations: List[nn.Module],
        temperature_start: float = 1.0,
        drop_path_prob_start: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.candidate_operations = nn.ModuleList(candidate_operations)
        self.temperature = temperature_start
        self.operations_count = len(candidate_operations)
        self.drop_path_prob = drop_path_prob_start

        self.architecture_parameters: nn.ParameterList

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the search space."""
        pass

    @abstractmethod
    def get_weights(self, node_idx: int, edge_idx: int) -> torch.Tensor:
        """Get normalized weights for operations on an edge."""
        pass

    def update_temperature(self, temperature: float):
        self.temperature = temperature

    def update_drop_path_prob(self, drop_path_prob: float):
        self.drop_path_prob = drop_path_prob

    def derive_architecture(self):
        """Derive the final architecture from the current architecture parameters.
        It can be overridden by the specific DARTS variant.
        """
        final_architecture = []
        for node in range(self.num_nodes):
            node_operations = []
            for i in range(node + 2):
                weights = self.get_weights(node, i)
                best_operation_index = int(weights.argmax().item())
                best_operation = self.candidate_operations[best_operation_index]
                node_operations.append((i, best_operation))
            final_architecture.append(node_operations)
        return final_architecture
