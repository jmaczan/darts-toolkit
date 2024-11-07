import os.path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.sgd import SGD

from base.operations import (
    AvgPool,
    DynamicSizeConv2d,
    DynamicSizeDilatedConv2d,
    DynamicSizeSeparableConv2d,
    Identity,
    MaxPool,
    ZeroOp,
)
from components.auxiliary_classifier import AuxiliaryHead
from components.regularization import DropPath
from components.schedulers import DropPathScheduler, TemperatureScheduler
from data.cifar_10 import CIFAR10DataModule
from default.classifier import get_default_classifier
from default.stem import get_default_stem
from utils.tensor import get_output_channels
from utils.yaml import load_config


class LPCDARTSSearchSpace(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_channels,
        num_partial_channel_connections,
        edge_norm_init=1.0,
        edge_norm_strength=1.0,
        num_segments=4,
        drop_path_prob_start=0.0,
        temperature_start=1.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.num_partial_channel_connections = num_partial_channel_connections
        self.edge_norm_init = edge_norm_init
        self.edge_norm_strength = edge_norm_strength
        self.num_segments = num_segments  # K in the PC-DARTS paper
        self.channels_per_segment = in_channels // num_segments
        self.channels_to_sample_per_segment = (
            num_partial_channel_connections // num_segments
        )
        self.drop_path = DropPath(drop_path_prob_start)
        self.temperature = temperature_start
        # set of possible candidates for operations
        self.candidate_operations = nn.ModuleList(
            [
                Identity(),  # skip connection
                DynamicSizeConv2d(kernel_size=3, padding=1),  # 3x3 conv
                DynamicSizeConv2d(kernel_size=5, padding=2),  # 5x5 conv
                DynamicSizeConv2d(kernel_size=7, padding=3),  # 7x7 conv
                DynamicSizeConv2d(kernel_size=1),  # 1x1 conv
                MaxPool(kernel_size=3, stride=1, padding=1),  # 3x3 max pool
                AvgPool(kernel_size=3, stride=1, padding=1),  # 3x3 avg pool
                DynamicSizeSeparableConv2d(
                    kernel_size=3, padding=1
                ),  # 3x3 separable conv
                DynamicSizeSeparableConv2d(
                    kernel_size=5, padding=2
                ),  # 5x5 separable conv
                DynamicSizeDilatedConv2d(
                    kernel_size=3, padding=2, dilation=2
                ),  # 3x3 dilated conv
                DynamicSizeDilatedConv2d(
                    kernel_size=5, padding=4, dilation=2
                ),  # 5x5 dilated conv
                ZeroOp(),  # zero operation (no connection)
            ]
        )

        # Get num_ops dynamically from candidate_operations
        self.num_ops = len(self.candidate_operations)

        # initial architecture params (alpha)
        self.arch_parameters = nn.ParameterList(
            [
                nn.Parameter(1e-3 * torch.randn(i + 2, self.num_ops))
                for i in range(self.num_nodes)
            ]
        )

        # edge normalization
        self.edge_norms = nn.ParameterList(
            [
                nn.Parameter(torch.full((i + 2,), edge_norm_init))
                for i in range(self.num_nodes)
            ]
        )

    def update_drop_path_prob(self, drop_path_prob: float):
        self.drop_path.drop_prob = drop_path_prob

    def update_temperature(self, temperature: float):
        self.temperature = temperature

    # this function is the heart of PC-DARTS, so for anyone reading this, let's break this down
    def forward(self, x):
        states = [x]  # this is the input to the cell
        for node in range(
            self.num_nodes
        ):  # iterates over each intermediate node in the cell
            node_inputs = []  # for each node, we will store inputs here
            for i in range(min(node + 2, len(states))):
                sampled_channels = []
                # previously, we sampled channels randomly from all channels
                # now, we sample channels from each segment separately systematically
                # divide channels into K segments, and sample equal number of channels from each segment
                # then, concatenate all channels from all segments
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

                clamped_norms = self.edge_norms[node][i].clamp(min=1e-5)
                normalized_weights = F.softmax(
                    self.arch_parameters[node][i]
                    / (clamped_norms**self.edge_norm_strength * self.temperature),
                    dim=-1,
                )  # softmax to architectural parameters for current node and input
                # these parametsr tell us about importance of each operation in this particular connection
                for j, op in enumerate(self.candidate_operations):
                    # go through all candidate operations (poolings, convs etc.)
                    if isinstance(
                        op,
                        (
                            DynamicSizeConv2d,
                            DynamicSizeSeparableConv2d,
                            DynamicSizeDilatedConv2d,
                            nn.Identity,
                        ),
                    ):
                        # for convolutions and identity, apply operation only on randomly sampled channels
                        x_sampled = states[i][:, sampled_channels]
                        output = op(x_sampled)

                        # expand output back to original amount of channels
                        expanded_output = torch.zeros_like(states[i])
                        expanded_output[:, sampled_channels] = output
                        node_inputs.append(
                            self.drop_path(normalized_weights[j] * expanded_output)
                        )
                    elif isinstance(op, (nn.MaxPool2d, nn.AvgPool2d)):
                        # pooling operations
                        node_inputs.append(
                            self.drop_path(normalized_weights[j] * op(states[i]))
                        )
                    elif isinstance(op, ZeroOp):
                        # zero operation - no connection
                        node_inputs.append(
                            self.drop_path(
                                normalized_weights[j] * torch.zeros_like(states[i])
                            )
                        )
                    else:
                        raise ValueError(f"Unknown operation type: {type(op)}")
            states.append(
                sum(node_inputs)
            )  # after processing all inputs and operations for a node, we sum the weighted inputs
            # the sum is output of the current node and we append it to states list
        return states[-1]  # we return the last state, which is the output of the cell


class LPCDARTSLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.stem = get_default_stem()

        stem_output_channels = get_output_channels(self.stem)

        self.search_space = LPCDARTSSearchSpace(
            in_channels=stem_output_channels,
            num_nodes=config["model"]["num_nodes"],
            num_partial_channel_connections=config["model"][
                "num_partial_channel_connections"
            ],
            edge_norm_init=config["model"].get("edge_norm_init", 1.0),
            edge_norm_strength=config["model"].get("edge_norm_strength", 1.0),
            num_segments=config["model"].get("num_segments", 4),
            drop_path_prob_start=config["training"].get("drop_path_prob_start", 0.0),
        )

        self.classifier = get_default_classifier(
            in_features=stem_output_channels,
            out_features=config["model"]["num_classes"],
        )

        self.arch_params = list(self.search_space.arch_parameters.parameters())
        self.edge_norm_params = list(self.search_space.edge_norms.parameters())
        self.weight_params = list(self.stem.parameters()) + list(
            self.classifier.parameters()
        )

        self.drop_path_scheduler = DropPathScheduler(
            drop_path_prob_start=config["model"].get("drop_path_prob_start", 0.0),
            drop_path_prob_end=config["model"].get("drop_path_prob_end", 0.3),
            epochs=config["training"]["max_epochs"],
        )

        self.temperature_scheduler = TemperatureScheduler(
            temperature_start=config["model"].get("temperature_start", 1.0),
            temperature_end=config["model"].get("temperature_end", 0.1),
            epochs=config["training"]["max_epochs"],
        )

        self.auxiliary_weight = config["model"].get("auxiliary_weight", 0.4)
        self.auxiliary_head = AuxiliaryHead(
            in_channels=stem_output_channels,
            num_classes=config["model"]["num_classes"],
        )

        self.automatic_optimization = False

    def forward(self, x):
        x = self.stem(x)

        aux_logits = None
        if self.training:
            aux_logits = self.auxiliary_head(x)

        x = self.search_space(x)
        logits = self.classifier(x)

        if self.training and aux_logits is not None:
            return logits, aux_logits

        return logits

    def on_train_epoch_start(self):
        self.search_space.update_drop_path_prob(
            self.drop_path_scheduler(self.current_epoch)
        )
        self.search_space.update_temperature(
            self.temperature_scheduler(self.current_epoch)
        )
        self.log("drop_path_prob", self.search_space.drop_path.drop_prob)
        self.log("temperature", self.search_space.temperature)

    # bilevel optimization
    def training_step(self, batch, batch_idx):
        optimizer_weights, optimizer_arch, optimizer_edge_norm = self.optimizers()  # type: ignore

        input_train, target_train = batch["train"]  # for updating network weights
        input_search, target_search = batch[
            "search"
        ]  # for updating architecture params

        # update architecture parameters "upper-level optimization"
        optimizer_arch.zero_grad()
        logits_arch, aux_logits_arch = self(input_search)
        loss_arch = F.cross_entropy(logits_arch, target_search)
        if aux_logits_arch is not None:
            aux_loss_arch = F.cross_entropy(aux_logits_arch, target_search)
            loss_arch += self.auxiliary_weight * aux_loss_arch
        self.manual_backward(loss_arch)
        optimizer_arch.step()

        # update edge normalization parameters; it helps to stabilize the search process, by balacing the importance of different edges;
        # sounds smart, but I'm unsure how and why it works, yet
        optimizer_edge_norm.zero_grad()
        logits_edge_norm, aux_logits_edge_norm = self(input_search)
        loss_edge_norm = F.cross_entropy(logits_edge_norm, target_search)
        if aux_logits_edge_norm is not None:
            aux_loss_edge_norm = F.cross_entropy(aux_logits_edge_norm, target_search)
            loss_edge_norm += self.auxiliary_weight * aux_loss_edge_norm
        self.manual_backward(loss_edge_norm)
        optimizer_edge_norm.step()

        # update network weights "lower-level optimization"
        optimizer_weights.zero_grad()
        logits_weights, aux_logits_weights = self(input_train)
        loss_weights = F.cross_entropy(logits_weights, target_train)
        if aux_logits_weights is not None:
            aux_loss_weights = F.cross_entropy(aux_logits_weights, target_train)
            loss_weights += self.auxiliary_weight * aux_loss_weights
        self.manual_backward(loss_weights)
        optimizer_weights.step()

        self.log("train_loss_weights", loss_weights)
        self.log("train_loss_arch", loss_arch)
        self.log("train_loss_edge_norm", loss_edge_norm)

        return {"loss": loss_weights}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return {"test_loss": loss, "test_acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        # optimizer for model's weights
        optimizer_weights = SGD(
            params=self.weight_params,
            lr=self.config["training"]["learning_rate"],
            momentum=self.config["training"]["momentum"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        # optimizer for architecture parameters
        optimizer_arch = Adam(
            params=self.arch_params,
            lr=self.config["training"]["arch_learning_rate"],
            betas=(0.5, 0.999),
            weight_decay=self.config["training"]["arch_weight_decay"],
        )

        # optimizer for edge normalization parameters
        optimizer_edge_norm = Adam(
            params=self.edge_norm_params,
            lr=self.config["training"]["edge_norm_learning_rate"],
            betas=(0.5, 0.999),
            weight_decay=self.config["training"]["edge_norm_weight_decay"],
        )

        weights_scheduler = CosineAnnealingLR(
            optimizer=optimizer_weights, T_max=self.config["training"]["max_epochs"]
        )

        return [optimizer_weights, optimizer_arch, optimizer_edge_norm], [
            weights_scheduler
        ]

    def derive_architecture(self):
        final_architecture = []
        for node in range(self.search_space.num_nodes):
            node_operations = []
            for i in range(node + 2):
                # pick the operation with the highest weight for each edge
                clamped_norms = self.search_space.edge_norms[node][i].clamp(min=1e-5)
                normalized_weights = F.softmax(
                    self.search_space.arch_parameters[node][i]
                    / (clamped_norms**self.search_space.edge_norm_strength),
                    dim=-1,
                )
                best_operation_index = int(normalized_weights.argmax().item())
                best_operation = self.search_space.candidate_operations[
                    best_operation_index
                ]
                node_operations.append((i, best_operation))
            final_architecture.append(node_operations)
        return final_architecture

    def train_derived_architecture(
        self, derived_architecture, train_loader, val_loader, epochs=100
    ):
        """Train the derived architecture from scratch"""
        derived_model = DerivedPCDARTSModel(
            config=self.config,
            derived_architecture=derived_architecture,
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=self.config["training"].get("accelerator", "auto"),
            devices=self.config["training"].get("devices", 1),
            callbacks=[ModelCheckpoint(monitor="val_acc", mode="max")],
            logger=TensorBoardLogger(
                save_dir=self.config["logging"]["log_dir"], name="derived_model"
            ),
        )

        trainer.fit(derived_model, train_loader, val_loader)

        return derived_model


class DerivedPCDARTSModel(pl.LightningModule):
    def __init__(self, derived_architecture, config):
        super().__init__()
        self.derived_architecture = derived_architecture
        self.config = config
        self.save_hyperparameters(ignore=["derived_architecture"])
        self.num_nodes = len(derived_architecture)

        self.in_channels = config["model"]["in_channels"]
        self.num_classes = config["model"]["num_classes"]
        self.num_cells = config["model"]["num_cells"]

        self.stem = get_default_stem()
        self.cell_channels = get_output_channels(self.stem)

        self.cells = nn.ModuleList([self._make_cell() for _ in range(self.num_cells)])

        self.classifier = get_default_classifier(
            in_features=self.cell_channels, out_features=self.num_classes
        )

        # Auxiliary classifier
        self.auxiliary_weight = config["model"].get("auxiliary_weight", 0.4)

        self.auxiliary_head_position = (
            self.num_cells // 3 * 2
        )  # put it at 2/3 of the network
        self.auxiliary_head = AuxiliaryHead(
            in_channels=self.cell_channels,
            num_classes=self.num_classes,
        )

    def _make_cell(self) -> nn.ModuleList:
        cell = nn.ModuleList()
        for node_ops in self.derived_architecture:
            node = nn.ModuleList()
            for i, op in node_ops:
                node.append(op.to_trainable(self.cell_channels))
            cell.append(node)
        return cell

    def forward(self, x):
        print(f"Input shape: {x.shape}")

        x = self.stem(x)
        print(f"After stem shape: {x.shape}")

        aux_logits = None
        for i, cell in enumerate(self.cells):
            cell_states = [x]
            for node in cell:  # type: ignore
                node_inputs = []
                for i, op in enumerate(node):
                    if i < len(cell_states):  # only use available previous states
                        node_inputs.append(op(cell_states[i]))
                node_output = sum(node_inputs)
                cell_states.append(node_output)
            x = cell_states[-1]  # use the latest state as cell output
            print(f"After cell {i} shape: {x.shape}")
            if self.training and i == self.auxiliary_head_position:
                aux_logits = self.auxiliary_head(x)

        print(f"Classifier input features: {self.classifier[-1].in_features}")
        print(f"Classifier output features: {self.classifier[-1].out_features}")
        logits = self.classifier(x)

        if self.training and aux_logits is not None:
            return logits, aux_logits
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)

        # Handle both cases: with and without auxiliary logits
        if isinstance(output, tuple):
            logits, aux_logits = output
            main_loss = F.cross_entropy(logits, y)
            aux_loss = F.cross_entropy(aux_logits, y)
            loss = main_loss + self.auxiliary_weight * aux_loss

            self.log("train_loss", loss)
            self.log("train_main_loss", main_loss)
            self.log("train_aux_loss", aux_loss)
        else:
            logits = output
            loss = F.cross_entropy(logits, y)
            self.log("train_loss", loss)

        acc = (logits.argmax(dim=1) == y).float().mean()
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

    def configure_optimizers(self):
        optimizer = SGD(
            params=self.parameters(),
            lr=self.config["training"]["learning_rate"],
            momentum=self.config["training"]["momentum"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config["training"]["max_epochs"]
        )
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"test_loss": loss, "test_acc": acc}


def main():
    config = load_config(os.path.join("src", "config.yaml"))
    data_module = CIFAR10DataModule(config)

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


if __name__ == "__main__":
    main()
