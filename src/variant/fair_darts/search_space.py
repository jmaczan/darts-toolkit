import torch
import torch.nn as nn

from base.search_space import BaseSearchSpace
from component.regularization import DropPath
from variant.fair_darts.default import default_fair_darts_candidate_operations


class FairDARTSSearchSpace(BaseSearchSpace):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        temperature_start: float = 1.0,
        drop_path_prob_start: float = 0.0,
        beta: float = 1.0,
        reg_strength: float = 0.1,
        candidate_operations=default_fair_darts_candidate_operations(),
        arch_parameters=None,
    ):
        super().__init__(
            num_nodes=num_nodes,
            in_channels=in_channels,
            candidate_operations=candidate_operations,
            temperature_start=temperature_start,
            drop_path_prob_start=drop_path_prob_start,
        )

        self.beta = beta
        self.reg_strength = reg_strength
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
        """Get sigmoid-based weights for operations on an edge."""
        return torch.sigmoid(self.beta * self.arch_parameters[node_idx][edge_idx])

    def compute_regularization_loss(self) -> torch.Tensor:
        """Compute the competition-aware regularization loss."""
        reg_loss = torch.tensor(
            0.0, device=self.arch_parameters[0].device
        )  # Initialize as tensor
        for node in range(self.num_nodes):
            for i in range(node + 2):
                weights = self.get_weights(node, i)
                correlation = torch.mm(weights.unsqueeze(1), weights.unsqueeze(0))
                correlation = correlation * (
                    1 - torch.eye(correlation.size(0), device=correlation.device)
                )
                reg_loss += correlation.sum()
        return self.reg_strength * reg_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [x]
        for node in range(self.num_nodes):
            node_inputs = []
            for i in range(min(node + 2, len(states))):
                operation_weights = self.get_weights(node, i)

                for j, op in enumerate(self.candidate_operations):
                    node_inputs.append(
                        self.drop_path(operation_weights[j] * op(states[i]))
                    )

            states.append(torch.sum(torch.stack(node_inputs), dim=0))

        return states[-1]
