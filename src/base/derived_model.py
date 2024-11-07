from typing import Any, Dict

import pytorch_lightning as pl
import torch


class BaseDerivedModel(pl.LightningModule):
    """Base class for derived DARTS models after architecture search."""

    def __init__(self, config: Dict[str, Any], derived_arch: list):
        super().__init__()
        self.config = config
        self.derived_arch = derived_arch

        self.stem = # same as for base model
