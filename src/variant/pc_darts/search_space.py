import torch
import torch.nn as nn
import torch.nn.functional as F

from base.operations import (
    DynamicSizeConv2d,
    DynamicSizeDilatedConv2d,
    DynamicSizeSeparableConv2d,
    Identity,
)
from base.search_space import BaseSearchSpace
from component.regularization import DropPath
from variant.pc_darts.default import default_pc_darts_candidate_operations


class PCDARTSSearchSpace(BaseSearchSpace):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        num_partial_channel_connections: int,
        edge_norm_init: float = 1.0,
        edge_norm_strength: float = 1.0,
        num_segments: int = 4,
        temperature_start: float = 1.0,
        drop_path_prob_start: float = 0.0,
        candidate_operations=default_pc_darts_candidate_operations(),
        arch_parameters=None,
    ):
        super().__init__(
            num_nodes=num_nodes,
            in_channels=in_channels,
            candidate_operations=candidate_operations,
            temperature_start=temperature_start,
            drop_path_prob_start=drop_path_prob_start,
        )

        self.num_partial_channel_connections = num_partial_channel_connections
        self.edge_norm_init = edge_norm_init
        self.edge_norm_strength = edge_norm_strength
        self.num_segments = num_segments
        self.channels_per_segment = in_channels // num_segments
        self.channels_to_sample_per_segment = (
            num_partial_channel_connections // num_segments
        )
        self.num_ops = len(self.candidate_operations)

        # initial architecture params (alpha)
        self.arch_parameters = arch_parameters or nn.ParameterList(
            [
                nn.Parameter(1e-3 * torch.randn(i + 2, self.num_ops))
                for i in range(self.num_nodes)
            ]
        )

        # Edge normalization parameters
        self.edge_norms = nn.ParameterList(
            [
                nn.Parameter(torch.full((i + 2,), edge_norm_init))
                for i in range(self.num_nodes)
            ]
        )

        self.drop_path = DropPath(drop_path_prob_start)

    def get_weights(self, node_idx: int, edge_idx: int) -> torch.Tensor:
        """Get normalized weights for operations on an edge."""
        clamped_norms = self.edge_norms[node_idx][edge_idx].clamp(min=1e-5)
        return F.softmax(
            self.arch_parameters[node_idx][edge_idx]
            / (clamped_norms**self.edge_norm_strength * self.temperature),
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [x]
        for node in range(self.num_nodes):
            node_inputs = []
            for i in range(min(node + 2, len(states))):
                sampled_channels = []

                # Sample channels from each segment
                for segment in range(self.num_segments):
                    segment_start = segment * self.channels_per_segment
                    segment_end = segment_start + self.channels_per_segment
                    segment_indices = torch.arange(
                        segment_start, segment_end, device=x.device
                    )
                    segment_samples = segment_indices[
                        torch.randperm(len(segment_indices))[
                            : self.channels_to_sample_per_segment
                        ]
                    ]
                    sampled_channels.append(segment_samples)
                sampled_channels = torch.cat(sampled_channels)

                normalized_weights = self.get_weights(node, i)

                for j, op in enumerate(self.candidate_operations):
                    if isinstance(
                        op,
                        (
                            DynamicSizeConv2d,
                            DynamicSizeSeparableConv2d,
                            DynamicSizeDilatedConv2d,
                            Identity,
                        ),
                    ):
                        x_sampled = states[i][:, sampled_channels]
                        output = op(x_sampled)
                        expanded_output = torch.zeros_like(states[i])
                        expanded_output[:, sampled_channels] = output
                        node_inputs.append(
                            self.drop_path(normalized_weights[j] * expanded_output)
                        )
                    else:
                        node_inputs.append(
                            self.drop_path(normalized_weights[j] * op(states[i]))
                        )

            states.append(
                torch.sum(torch.stack(node_inputs), dim=0)
                if node_inputs
                else x.new_zeros(x.shape)
            )

        return states[-1]
