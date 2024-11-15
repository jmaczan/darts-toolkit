import torch
import torch.nn as nn
import torch.nn.functional as F

from base.search_space import BaseSearchSpace
from component.regularization import DropPath
from variant.darts.default import default_darts_candidate_operations


class DARTSSearchSpace(BaseSearchSpace):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        temperature_start: float = 1.0,
        drop_path_prob_start: float = 0.0,
        candidate_operations=default_darts_candidate_operations(),
        arch_parameters=None,
    ):
        super().__init__(
            num_nodes=num_nodes,
            in_channels=in_channels,
            candidate_operations=candidate_operations,
            temperature_start=temperature_start,
            drop_path_prob_start=drop_path_prob_start,
        )

        self.num_ops = len(self.candidate_operations)
        self.drop_path = DropPath(drop_path_prob_start)

        # Initialize architecture parameters
        self.arch_parameters = arch_parameters or nn.ParameterList(
            [
                nn.Parameter(1e-3 * torch.randn(i + 2, self.num_ops))
                for i in range(self.num_nodes)
            ]
        )

    def get_weights(self, node_idx: int, edge_idx: int) -> torch.Tensor:
        """Get normalized weights for operations on an edge."""
        return F.softmax(
            self.arch_parameters[node_idx][edge_idx] / self.temperature,
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [x]
        for node in range(self.num_nodes):
            node_inputs = []
            for i in range(min(node + 2, len(states))):
                normalized_weights = self.get_weights(node, i)

                for j, op in enumerate(self.candidate_operations):
                    node_inputs.append(
                        self.drop_path(normalized_weights[j] * op(states[i]))
                    )

            states.append(
                torch.sum(torch.stack(node_inputs), dim=0)
                if node_inputs
                else x.new_zeros(x.shape)
            )

        return states[-1]
