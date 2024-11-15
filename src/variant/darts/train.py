import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from base.derived_model import BaseDerivedModel
from datamodule.cifar_10 import CIFAR10DataModule
from utils.yaml import load_config
from variant.darts.module import DARTSModule


def train(config_path: str = os.path.join("src", "config.yaml")):
    config = load_config(config_path=config_path)
    data_module = CIFAR10DataModule(config)

    # Search phase
    search_model = DARTSModule(config)
    search_trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="gpu" if config["training"].get("gpus") else "auto",
        devices=config["training"].get("gpus") or "auto",
        callbacks=[RichProgressBar()],
        logger=TensorBoardLogger(
            config["logging"]["log_dir"],
            name=f"{config['logging']['experiment_name']}_search",
        ),
    )

    # Train the search model
    search_trainer.fit(search_model, data_module)

    # Test the search model
    search_trainer.test(search_model, datamodule=data_module)

    # Derive and train the final architecture
    derived_architecture = search_model.derive_architecture()
    derived_model = BaseDerivedModel(
        derived_architecture=derived_architecture, config=config
    )

    derived_trainer = pl.Trainer(
        max_epochs=config["training"]["derived_epochs"],
        accelerator="gpu" if config["training"].get("gpus") else "auto",
        devices=config["training"].get("gpus") or "auto",
        callbacks=[ModelCheckpoint(monitor="val_acc", mode="max"), RichProgressBar()],
        logger=TensorBoardLogger(
            config["logging"]["log_dir"],
            name=f"{config['logging']['experiment_name']}_derived",
        ),
    )

    derived_trainer.fit(
        derived_model,
        train_dataloaders=data_module.train_dataloader()["train"],
        val_dataloaders=data_module.val_dataloader(),
    )

    derived_trainer.test(derived_model, datamodule=data_module)


if __name__ == "__main__":
    train()
