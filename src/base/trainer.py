import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from base.darts_model import BaseDARTSModel
from base.derived_model import BaseDerivedModel


class DARTSTrainer:
    """Trainer for DARTS variants.
    Trains both search and derived models.
    """

    def __init__(self, config: dict):
        self.config = config

    def train_search(self, model: BaseDARTSModel, datamodule):
        trainer = pl.Trainer(
            max_epochs=self.config["training"]["max_epochs"],
            accelerator=self.config["training"].get("accelerator", "auto"),
            devices=self.config["training"].get("devices", 1),
            callbacks=[
                ModelCheckpoint(monitor="val_acc", mode="max"),
                RichProgressBar(),
            ],
            logger=TensorBoardLogger(
                self.config["logging"]["log_dir"],
                name=f"{self.config['logging']['experiment_name']}_search",
            ),
        )
        trainer.fit(model, datamodule)
        return trainer.test(model, datamodule=datamodule)

    def train_derived(self, model: BaseDerivedModel, datamodule):
        trainer = pl.Trainer(
            max_epochs=self.config["training"]["derived_epochs"],
            accelerator=self.config["training"].get("accelerator", "auto"),
            devices=self.config["training"].get("devices", 1),
            callbacks=[
                ModelCheckpoint(monitor="val_acc", mode="max"),
                RichProgressBar(),
            ],
            logger=TensorBoardLogger(
                self.config["logging"]["log_dir"],
                name=f"{self.config['logging']['experiment_name']}_derived",
            ),
        )
        trainer.fit(model, datamodule)
        return trainer.test(model, datamodule=datamodule)
