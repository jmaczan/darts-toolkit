import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import networkx as nx
import matplotlib.pyplot as plt


class ZeroOp(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class LPCDARTSSearchSpace(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_channels,
        num_partial_channel_connections,
        edge_norm_init=1.0,
        edge_norm_strength=1.0,
    ):
        super(LPCDARTSSearchSpace, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.num_partial_channel_connections = num_partial_channel_connections
        self.edge_norm_init = edge_norm_init
        self.edge_norm_strength = edge_norm_strength

        # set of possible candidates for operations
        self.candidate_operations = nn.ModuleList(
            [
                nn.Identity(),  # skip connection
                DynamicSizeConv2d(kernel_size=3, padding=1),  # 3x3 conv
                DynamicSizeConv2d(kernel_size=5, padding=2),  # 5x5 conv
                DynamicSizeConv2d(kernel_size=7, padding=3),  # 7x7 conv
                DynamicSizeConv2d(kernel_size=1),  # 1x1 conv
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 3x3 max pool
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),  # 3x3 avg pool
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
                nn.Parameter(torch.randn(i + 2, self.num_ops))
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

    # this function is the heart of PC-DARTS, so for anyone reading this, let's break this down
    def forward(self, x):
        states = [x]  # this is the input to the cell
        for node in range(
            self.num_nodes
        ):  # iterates over each intermediate node in the cell
            node_inputs = []  # for each node, we will store inputs here
            for i in range(min(node + 2, len(states))):
                sampled_channels = torch.randperm(self.in_channels)[
                    : self.num_partial_channel_connections
                ]
                # partial channel connections - reduced number of channels, randomly chosen, in search space
                # iterate over all possible input states for current node.
                # +2 because each node can take input from all previous nodes plus two initial inputs
                # which is output of the previous call and output of the previous-previous cell

                clamped_norms = self.edge_norms[node][i].clamp(min=1e-5)
                normalized_weights = F.softmax(
                    self.arch_parameters[node][i]
                    / (clamped_norms**self.edge_norm_strength),
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
                        node_inputs.append(normalized_weights[j] * expanded_output)
                    elif isinstance(op, (nn.MaxPool2d, nn.AvgPool2d)):
                        # pooling operations
                        node_inputs.append(normalized_weights[j] * op(states[i]))
                    elif isinstance(op, ZeroOp):
                        # zero operation - no connection
                        node_inputs.append(
                            torch.zeros_like(states[i]) * normalized_weights[j]
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
        super(LPCDARTSLightningModule, self).__init__()
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

        self.automatic_optimization = False

    def forward(self, x):
        x = self.stem(x)
        x = self.search_space(x)
        return self.classifier(x)

    # bilevel optimization
    def training_step(self, batch, batch_idx):
        optimizer_weights, optimizer_arch, optimizer_edge_norm = self.optimizers()

        input_train, target_train = batch["train"]  # for updating network weights
        input_search, target_search = batch[
            "search"
        ]  # for updating architecture params

        # update architecture parameters "upper-level optimization"
        optimizer_arch.zero_grad()
        logits_arch = self(input_search)
        loss_arch = F.cross_entropy(logits_arch, target_search)
        self.manual_backward(loss_arch)
        optimizer_arch.step()

        # update edge normalization parameters; it helps to stabilize the search process, by balacing the importance of different edges;
        # sounds smart, but I'm unsure how and why it works, yet
        optimizer_edge_norm.zero_grad()
        logits_edge_norm = self(input_search)
        loss_edge_norm = F.cross_entropy(logits_edge_norm, target_search)
        self.manual_backward(loss_edge_norm)
        optimizer_edge_norm.step()

        # update network weights "lower-level optimization"
        optimizer_weights.zero_grad()
        logits_weights = self(input_train)
        loss_weights = F.cross_entropy(logits_weights, target_train)
        self.manual_backward(loss_weights)
        optimizer_weights.step()

        self.log("train_loss_weights", loss_weights)
        self.log("train_loss_arch", loss_arch)
        self.log("train_loss_edge_norm", loss_edge_norm)

        return {"loss": loss_weights}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        # optimizer for model's weights
        optimizer_weights = torch.optim.SGD(
            params=self.weight_params,
            lr=self.config["training"]["learning_rate"],
            momentum=self.config["training"]["momentum"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        # optimizer for architecture parameters
        optimizer_arch = torch.optim.Adam(
            params=self.arch_params,
            lr=self.config["training"]["arch_learning_rate"],
            betas=(0.5, 0.999),
            weight_decay=self.config["training"]["arch_weight_decay"],
        )

        # optimizer for edge normalization parameters
        optimizer_edge_norm = torch.optim.Adam(
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
                best_operation_index = normalized_weights.argmax().item()
                best_operation_type = type(
                    self.search_space.candidate_operations[best_operation_index]
                )
                node_operations.append((i, best_operation_type))
            final_architecture.append(node_operations)
        return final_architecture

    def train_derived_architecture(
        self, derived_architecture, train_loader, val_loader, epochs=100
    ):
        """Train the derived architecture from scratch"""
        derived_model = DerivedPCDARTSModel(
            derived_architecture=derived_architecture,
            num_classes=self.config["model"]["num_classes"],
            in_channels=self.search_space.in_channels,
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

    def visualize_architecture(self):
        G = nx.DiGraph()
        G.add_node("input")

        for i, node_ops in enumerate(self.derive_architecture):
            node_name = f"node_{i}"
            G.add_node(node_name)

            for j, (input_idx, op_type) in enumerate(node_ops):
                if input_idx == 0:
                    G.add_edge("input", node_name, operation=op_type.__name__)
                else:
                    G.add_edge(
                        f"node_{input_idx-1}", node_name, operation=op_type.__name__
                    )

        G.add_edge("output")
        G.add_edge(f"node_{len(self.derive_architecture) - 1}", "output")

        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=1500,
            font_size=10,
            arrows=True,
        )
        edge_labels = nx.get_edge_attributes(G, "operation")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("Derived architecture")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config["data"]["data_dir"]
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]

        self.transform = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),  # CIFAR10 - see https://github.com/kuangliu/pytorch-cifar/issues/8
            ]
        )

        # transforms without data augmentation
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = datasets.CIFAR10(
                root=self.data_dir, train=True, transform=self.transform
            )
            train_size = int(0.8 * len(cifar_full))
            val_size = len(cifar_full) - train_size

            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [train_size, val_size]
            )

            search_size = int(0.5 * len(self.cifar_train))
            self.cifar_train, self.cifar_search = random_split(
                self.cifar_train, [len(self.cifar_train) - search_size, search_size]
            )

        if stage == "test" or stage is None:
            self.cifar_test = datasets.CIFAR10(
                root=self.data_dir, train=False, transform=self.test_transform
            )

    def train_dataloader(self):
        return {
            "train": DataLoader(
                self.cifar_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            ),
            "search": DataLoader(
                self.cifar_search,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            ),
        }

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class DynamicSizeConv2d(nn.Module):
    def __init__(self, kernel_size, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x):
        weight = torch.randn(
            x.size(1), x.size(1), self.kernel_size, self.kernel_size, device=x.device
        )

        return F.conv2d(x, weight, padding=self.padding)


class DynamicSizeSeparableConv2d(nn.Module):
    def __init__(self, kernel_size, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x):
        # Depthwise convolution
        depthwise_weight = torch.randn(
            x.size(1), 1, self.kernel_size, self.kernel_size, device=x.device
        )

        depthwise_output = F.conv2d(
            x, depthwise_weight, padding=self.padding, groups=x.size(1)
        )

        # Pointwise convolution
        pointwise_weight = torch.randn(x.size(1), x.size(1), 1, 1, device=x.device)

        return F.conv2d(depthwise_output, pointwise_weight, padding=0)


class DynamicSizeDilatedConv2d(nn.Module):
    def __init__(self, kernel_size, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        weight = torch.randn(
            x.size(1), x.size(1), self.kernel_size, self.kernel_size, device=x.device
        )

        return F.conv2d(x, weight, padding=self.padding, dilation=self.dilation)


class DerivedPCDARTSModel(pl.LightningModule):
    def __init__(self, derived_architecture, config):
        super().__init__()
        self.derived_architecture = derived_architecture
        self.save_hyperparameters()
        self.num_nodes = len(derived_architecture)

        self.in_channels = config["model"]["in_channels"]
        self.num_classes = config["model"]["num_classes"]
        self.num_cells = config["model"]["num_cells"]

        self.stem = get_default_stem()

        self.cells = nn.ModuleList([self._make_cell() for _ in range(self.num_cells)])
        self.global_pooling = nn.AdaptiveAvgPool2d(output_size=1)

        stem_output_channels = get_output_channels(self.stem)

        self.classifier = get_default_classifier(
            in_features=stem_output_channels, out_features=self.num_classes
        )

    def _make_cell(self):
        cell = nn.ModuleList()

        cell_channels = self.in_channels * (2 ** len((self.cells)))

        for node_ops in self.derived_architecture:
            node = nn.ModuleList()
            for i, op_type in node_ops:
                if op_type == nn.Identity:
                    node.append(op_type())
                elif op_type == ZeroOp:
                    node.append(op_type())
                elif op_type in (nn.MaxPool2d, nn.AvgPool2d):
                    node.append(op_type(kernel_size=3, stride=1, padding=1))
                elif op_type == DynamicSizeConv2d:
                    conv = nn.Conv2d(
                        in_channels=cell_channels,
                        out_channels=cell_channels,
                        kernel_size=op_type.kernel_size,
                        padding=op_type.padding,
                        bias=False,
                    )
                    bn = nn.BatchNorm2d(num_features=cell_channels)
                    node.append(nn.Sequential(conv, bn, nn.ReLU(inplace=True)))
                elif op_type == DynamicSizeSeparableConv2d:
                    depthwise = nn.Conv2d(
                        in_channels=cell_channels,
                        out_channels=cell_channels,
                        kernel_size=op_type.kernel_size,
                        padding=op_type.padding,
                        groups=cell_channels,
                        bias=False,
                    )
                    pointwise = nn.Conv2d(
                        in_channels=cell_channels,
                        out_channels=cell_channels,
                        kernel_size=1,
                        bias=False,
                    )
                    bn = nn.BatchNorm2d(num_features=cell_channels)
                    node.append(
                        nn.Sequential(depthwise, pointwise, bn, nn.ReLU(inplace=True))
                    )
                elif op_type == DynamicSizeDilatedConv2d:
                    conv = nn.Conv2d(
                        in_channels=cell_channels,
                        out_channels=cell_channels,
                        kernel_size=op_type.kernel_size,
                        padding=op_type.padding,
                        dilation=op_type.dilation,
                        bias=False,
                    )
                    bn = nn.BatchNorm2d(num_features=cell_channels)
                    node.append(nn.Sequential(conv, bn, nn.ReLU(inplace=True)))
            cell.append(node)

        return cell

    def forward(self, x):
        x = self.stem(x)

        for cell in self.cells:
            cell_states = [x]
            for node in cell:
                node_inputs = []
                for i, op in enumerate(node):
                    if i < len(cell_states):  # only use available previous states
                        node_inputs.append(op(cell_states[i]))
                node_output = sum(node_inputs)
                cell_states.append(node_output)
            x = cell_states[-1]  # use the latest state as cell output

        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
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
        optimizer = torch.optim.SGD(
            params=self.parameters(),
            lr=self.config["training"]["learning_rate"],
            momentum=self.config["training"]["momentum"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config["training"]["max_epochs"]
        )
        return [optimizer], [scheduler]


def get_default_stem():
    return nn.Sequential(
        nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(num_features=32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
    )


def get_default_classifier(in_features, out_features):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(),
        nn.Linear(
            in_features=in_features,
            out_features=out_features,
        ),
    )


def get_output_channels(module):
    if hasattr(module, "num_features"):
        return module.num_features
    elif hasattr(module, "out_channels"):
        return module.out_channels
    elif isinstance(module, nn.Sequential):
        for layer in reversed(module):
            if hasattr(layer, "num_features"):
                return layer.num_features
            if hasattr(layer, "out_channels"):
                return layer.out_channels
    else:
        raise ValueError(
            f"Unsupported module type: {type(module)}. Perhaps you want to add it to _get_output_channel() function to compute number of channels correctly"
        )


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


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
