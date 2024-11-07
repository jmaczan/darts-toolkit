# `darts-toolkit`

Differentiable Architecture Search Toolkit in PyTorch Lightning

> [!TIP]
> Boost your research and use solid engineering practices out-of-the-box

Use this toolkit to:

- Research your own DARTS algorithm with pre-built components and create your own components
- Use existing DARTS architectures, like [Partially-Connected](https://arxiv.org/abs/1907.05737) [Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)
- Configure hyperparameters with `yaml` files
- Scale to multiple GPUs with no effort
- Visualize your neural network architecture

## Examples

#### Find a network architecture for image recognition

```py
from darts_toolkit.models import LPCDARTSLightningModule
from darts_toolkit.data import CIFAR10DataModule
from darts_toolkit.utils.yaml import load_config
import yaml

# Load configuration
config = load_config(os.path.join("src", "config.yaml"))

# Create data module
data_module = CIFAR10DataModule(config)

# Create model
model = LPCDARTSLightningModule(config)

# Search phase
search_model = LPCDARTSLightningModule(config)
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
```

#### Train a derived architecture

```py
# Derive and train the final architecture
derived_architecture = search_model.derive_architecture()
derived_model = DerivedPCDARTSModel(
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

# Train the derived model
derived_trainer.fit(
    derived_model,
    train_dataloaders=data_module.train_dataloader()["train"],
    val_dataloaders=data_module.val_dataloader(),
)

# Test the derived model
derived_trainer.test(derived_model, datamodule=data_module)
```

## Install

Using pip:

```sh
pip install git+https://github.com/jmaczan/darts-toolkit.git
```

Using uv:

```sh
uv pip install git+https://github.com/jmaczan/darts-toolkit.git
```

## Install (for development)

```sh
git clone https://github.com/jmaczan/darts-toolkit.git
cd darts-toolkit

# Install using uv (recommended)
uv pip install -e .

# Or install using pip
pip install -e .
```

## Prerequisities

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for package management

Also, it uses [Ruff](https://docs.astral.sh/ruff/installation/) for formatting if you run the project in VS Code. You can install Ruff plugin by Astral Software from extensions marketplace and you're good to go

```sh
uv sync
```

## Run

```sh
uv run python -m src.models.lightning_pc_darts
```

## Cite

If you use this software in your research, please use the following citation:

```bibtex
@software{Maczan_PCDARTS_2024,
author = {Maczan, Jędrzej Paweł},
title = {Differentiable Architecture Search Toolkit in PyTorch Lightning},
url = {https://github.com/jmaczan/darts-toolkit},
year = {2024},
publisher = {GitHub}
}
```

## License

GNU GPLv3

## Author

Jędrzej Maczan, 2024
