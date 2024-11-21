from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.cell import Cell
from component.auxiliary_classifier import AuxiliaryHead
from default.classifier import get_default_classifier
from default.optimizers import (
    get_default_weights_optimizer,
)
from default.scheduler import get_default_weights_scheduler
from default.stem import get_default_stem
from utils.tensor import get_output_channels


class BaseDerivedModel(pl.LightningModule):
    """Base class for derived DARTS models after architecture search."""

    def __init__(
        self,
        config: Dict[str, Any],
        derived_architecture: list,
        stem=get_default_stem(),
        classifier=None,
        features={
            "auxiliary_head": False,
        },
        weights_optimizer=None,
        schedulers=None,
    ):
        super().__init__()
        self.config = config
        self.derived_architecture = derived_architecture
        self.auxiliary_head_position = None

        self.stem = stem
        self.stem_output_channels = get_output_channels(self.stem)
        self.classifier = classifier or get_default_classifier(
            in_features=self.stem_output_channels,
            out_features=config["model"]["num_classes"],
        )
        self.auxiliary_head = None
        self.features = features

        if self.features.get("auxiliary_head"):
            self.auxiliary_head = AuxiliaryHead(
                in_channels=self.stem_output_channels,  # type: ignore
                num_classes=config["model"]["num_classes"],
            )
            self.auxiliary_head_position = (
                self.config["model"]["num_cells"] // 3 * 2
            )  # put it at 2/3 of the network

        self.cells = self._build_cells()

        self.weight_params = list(self.stem.parameters()) + list(
            self.classifier.parameters()
        )

        if self.auxiliary_head:
            self.weight_params += list(self.auxiliary_head.parameters())

        self.weights_optimizer = weights_optimizer or get_default_weights_optimizer(
            self.weight_params, self.config
        )

        self.schedulers = (
            schedulers
            if (schedulers is not None and len(schedulers) > 0)
            else [get_default_weights_scheduler(self.weights_optimizer, self.config)]
        )

    def _build_cells(self) -> nn.ModuleList:
        cells = nn.ModuleList()
        num_cells = self.config["model"]["num_cells"]

        for cell_idx in range(num_cells):
            # Determine if this is a reduction cell (1/3 and 2/3 through the network)
            is_reduction = cell_idx in [num_cells // 3, 2 * num_cells // 3]

            if is_reduction:
                stride = 2
                # Double channels after reduction
                self.stem_output_channels *= 2  # type: ignore
            else:
                stride = 1

            cell = Cell(
                derived_architecture=self.derived_architecture,
                in_channels=self.stem_output_channels,  # type: ignore
                stride=stride,
            )
            cells.append(cell)

        return cells

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial stem processing
        x = self.stem(x)  # Shape: [B, 64, H, W]

        # Initialize previous cell states
        s0 = s1 = x  # Both start with stem output

        # Process each cell
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)  # Cell takes two previous states

            if (
                self.training
                and self.auxiliary_head
                and i == self.auxiliary_head_position
            ):
                aux_logits = self.auxiliary_head(s1)

        # Final classification
        out = self.adaptive_avg_pool2d(s1, 1)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        if self.training and self.auxiliary_head:
            return logits, aux_logits  # type: ignore
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)

        if isinstance(output, tuple):
            logits, aux_logits = output
            loss = F.cross_entropy(logits, y)
            if aux_logits is not None:
                loss += self.config["model"]["auxiliary_weight"] * F.cross_entropy(
                    aux_logits, y
                )
        else:
            logits = output
            loss = F.cross_entropy(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        return [
            self.weights_optimizer,
        ], self.schedulers
