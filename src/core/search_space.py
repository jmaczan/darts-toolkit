from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class SearchSpace(ABC, nn.Module):
    """
    Abstract base class for all DARTS search spaces.
    Provides the basic interface that all DARTS variants should implement.
    """

    def __init__(
        self,
        nodes_count: int,
        in_channels: int,
        candidate_operations: List[nn.Module],
        temperature: float = 1.0,
    ):
        super().__init__()
        self.nodes_count = nodes_count
        self.in_channels = in_channels
        self.candidate_operations = nn.ModuleList(candidate_operations)
        self.temperature = temperature
        self.operations_count = len(candidate_operations)

        self.architecture_parameters: nn.ParameterList
        self._initialize_architecture_parameters()

    @abstractmethod
    def _initialize_architecture_parameters(self):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_genotype(self) -> Dict[str, Any]:
        pass

    def update_temperature(self, temperature: float):
        self.temperature = temperature


class MixedOperation(nn.Module):
    """
    A mixed operation that combines multiple candidate operations.
    Base implementation that can be extended by different DARTS variants.
    """

    def __init__(self, operations: List[nn.Module], in_channels: int):
        super().__init__()
        self.operations = nn.ModuleList(operations)
        self.in_channels = in_channels
