import torch.nn.functional as F

from base.darts_model import BaseDARTSModel
from component.schedulers import DropPathScheduler
from variant.fair_darts.search_space import FairDARTSSearchSpace


class FairDARTSModule(BaseDARTSModel):
    def __init__(self, config: dict):
        search_space = FairDARTSSearchSpace(
            num_nodes=config["model"]["num_nodes"],
            in_channels=config["model"]["in_channels"],
            temperature_start=config["model"].get("temperature_start", 1.0),
            drop_path_prob_start=config["training"].get("drop_path_prob_start", 0.0),
            beta=config["model"].get("beta", 1.0),
            reg_strength=config["model"].get("reg_strength", 0.1),
        )

        super().__init__(
            search_space=search_space,
            config=config,
            features={"auxiliary_head": True},
        )

        self.drop_path_scheduler = DropPathScheduler(
            drop_path_prob_start=config["training"].get("drop_path_prob_start", 0.0),
            drop_path_prob_end=config["training"].get("drop_path_prob_end", 0.3),
            epochs=config["training"]["max_epochs"],
        )

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        optimizer_weights, optimizer_arch = self.optimizers()  # type: ignore

        input_train, target_train = batch["train"]
        input_search, target_search = batch["search"]

        # First update weights
        optimizer_weights.zero_grad()
        logits_weights, aux_logits_weights = self(input_train)
        loss_weights = F.cross_entropy(logits_weights, target_train)
        if aux_logits_weights is not None:
            aux_loss_weights = F.cross_entropy(aux_logits_weights, target_train)
            loss_weights += self.config["model"]["auxiliary_weight"] * aux_loss_weights

        self.manual_backward(loss_weights)
        optimizer_weights.step()

        # Then update architecture parameters
        optimizer_arch.zero_grad()
        logits_arch, aux_logits_arch = self(input_search)

        loss_arch = F.cross_entropy(logits_arch, target_search)
        if aux_logits_arch is not None:
            aux_loss_arch = F.cross_entropy(aux_logits_arch, target_search)
            loss_arch += self.config["model"]["auxiliary_weight"] * aux_loss_arch

        # Add FAIR competition-aware regularization
        reg_loss = self.search_space.compute_regularization_loss()
        loss_arch += reg_loss

        self.manual_backward(loss_arch)
        optimizer_arch.step()

        # Compute accuracies
        train_acc = (logits_weights.argmax(dim=-1) == target_train).float().mean()
        arch_acc = (logits_arch.argmax(dim=-1) == target_search).float().mean()

        # Log metrics
        self.log_dict(
            {
                "train_loss_weights": loss_weights.item(),
                "train_loss_arch": loss_arch.item(),
                "reg_loss": reg_loss.item(),
                "train_acc": train_acc.item(),
                "arch_acc": arch_acc.item(),
            },
            prog_bar=True,
        )

        return {"loss": loss_weights}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()

        self.log_dict(
            {
                "val_loss": loss.item(),
                "val_acc": acc.item(),
            },
            prog_bar=True,
        )

        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()

        self.log_dict(
            {
                "test_loss": loss.item(),
                "test_acc": acc.item(),
            },
            prog_bar=True,
        )

        return {"test_loss": loss, "test_acc": acc}

    def derive_architecture(self):
        """Expose the search space's derive_architecture method."""
        return self.search_space.derive_architecture()

    def on_train_epoch_start(self):
        # Update drop path probability
        self.search_space.update_drop_path_prob(
            self.drop_path_scheduler(self.current_epoch)
        )
        self.log("drop_path_prob", self.search_space.drop_path_prob)

    def configure_optimizers(self):
        """Configure optimizers with proper parameter groups."""
        # Weight parameters (excluding architecture parameters)
        weight_params = [
            p
            for n, p in self.named_parameters()
            if "arch_parameters" not in n and p.requires_grad
        ]

        # Architecture parameters
        arch_params = [
            p
            for n, p in self.named_parameters()
            if "arch_parameters" in n and p.requires_grad
        ]

        # Create optimizers with correct parameters
        optimizer_weights = torch.optim.SGD(
            weight_params,
            lr=self.config["training"]["learning_rate"],
            momentum=self.config["training"]["momentum"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        optimizer_arch = torch.optim.Adam(
            arch_params,
            lr=self.config["training"]["arch_learning_rate"],
            betas=(0.5, 0.999),
            weight_decay=self.config["training"]["arch_weight_decay"],
        )

        return [optimizer_weights, optimizer_arch]
