from abc import ABC, abstractmethod
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from base.search_space import BaseSearchSpace
from component.auxiliary_classifier import AuxiliaryHead
from default.classifier import get_default_classifier
from default.optimizers import (
    get_default_arch_optimizer,
    get_default_edge_norm_optimizer,
    get_default_weights_optimizer,
)
from default.scheduler import get_default_weights_scheduler
from default.stem import get_default_stem
from utils.tensor import get_output_channels


class BaseDARTSModel(pl.LightningModule, ABC):
    """Base class for all DARTS variants."""

    def __init__(
        self,
        search_space: BaseSearchSpace,
        config: Dict[str, Any],
        stem=get_default_stem(),
        classifier=None,
        features={
            "auxiliary_head": False,
            "edge_norm": False,
        },
        weights_optimizer=None,
        arch_optimizer=None,
        edge_norm_optimizer=None,
        schedulers=None,
    ):
        super().__init__()
        self.search_space = search_space
        self.config = config
        self.stem = stem
        self.stem_output_channels = get_output_channels(self.stem)
        self.classifier = classifier or get_default_classifier(
            in_features=self.stem_output_channels,
            out_features=config["model"]["num_classes"],
        )
        self.features = features
        self.auxiliary_head = None

        if self.features.get("auxiliary_head"):
            self.auxiliary_head = AuxiliaryHead(
                in_channels=self.stem_output_channels,  # type: ignore
                num_classes=config["model"]["num_classes"],
            )

        self.weight_params = list(self.stem.parameters()) + list(
            self.classifier.parameters()
        )

        if self.auxiliary_head:
            self.weight_params += list(self.auxiliary_head.parameters())

        self.arch_params = list(self.search_space.arch_parameters.parameters())

        if self.features.get("edge_norm"):
            self.edge_norm_params = (
                list(self.search_space.edge_norms.parameters())
                if hasattr(self.search_space, "edge_norms")
                else []
            )

        self.weights_optimizer = weights_optimizer or get_default_weights_optimizer(
            self.weight_params, self.config
        )

        self.arch_optimizer = arch_optimizer or get_default_arch_optimizer(
            self.arch_params, self.config
        )

        if self.features.get("edge_norm"):
            self.edge_norm_optimizer = (
                edge_norm_optimizer
                or get_default_edge_norm_optimizer(self.edge_norm_params, self.config)
            )

        self.schedulers = (
            schedulers
            if (schedulers is not None and len(schedulers) > 0)
            else [get_default_weights_scheduler(self.weights_optimizer, self.config)]
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        # Get intermediate output for auxiliary head
        features = self.search_space(x)

        # Apply auxiliary head before final classifier if in training mode
        aux_logits = None
        if self.auxiliary_head is not None and self.training:
            # Ensure features has the correct shape [B, C, H, W]
            if len(features.shape) != 4:
                raise ValueError(
                    f"Expected 4D tensor for auxiliary head, got shape {features.shape}"
                )
            aux_logits = self.auxiliary_head(features)

        # Apply final classifier
        logits = self.classifier(features)

        if aux_logits is not None:
            return logits, aux_logits
        return logits

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Implement specific DARTS variant training step."""
        pass

    def validation_step(
        self, batch: Any, batch_idx: int
    ) -> dict[str, float | torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [
            self.weights_optimizer,
            self.arch_optimizer,
        ], self.schedulers
