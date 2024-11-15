import torch.nn.functional as F

from base.darts_model import BaseDARTSModel
from component.schedulers import DropPathScheduler, TemperatureScheduler
from variant.darts.search_space import DARTSSearchSpace


class DARTSModule(BaseDARTSModel):
    def __init__(self, config: dict):
        search_space = DARTSSearchSpace(
            num_nodes=config["model"]["num_nodes"],
            in_channels=config["model"]["in_channels"],
            temperature_start=config["model"].get("temperature_start", 1.0),
            drop_path_prob_start=config["training"].get("drop_path_prob_start", 0.0),
        )

        super().__init__(
            search_space=search_space,
            config=config,
            features={"auxiliary_head": True},
        )

        # Initialize schedulers
        self.drop_path_scheduler = DropPathScheduler(
            drop_path_prob_start=config["training"].get("drop_path_prob_start", 0.0),
            drop_path_prob_end=config["training"].get("drop_path_prob_end", 0.3),
            epochs=config["training"]["max_epochs"],
        )

        self.temperature_scheduler = TemperatureScheduler(
            temperature_start=config["model"].get("temperature_start", 1.0),
            temperature_end=config["model"].get("temperature_end", 0.1),
            epochs=config["training"]["max_epochs"],
        )

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        optimizer_weights, optimizer_arch = self.optimizers()  # type: ignore

        input_train, target_train = batch["train"]
        input_search, target_search = batch["search"]

        # Update architecture parameters
        optimizer_arch.zero_grad()
        logits_arch, aux_logits_arch = self(input_search)
        loss_arch = F.cross_entropy(logits_arch, target_search)
        if aux_logits_arch is not None:
            aux_loss_arch = F.cross_entropy(aux_logits_arch, target_search)
            loss_arch += self.config["model"]["auxiliary_weight"] * aux_loss_arch
        self.manual_backward(loss_arch)
        optimizer_arch.step()

        # Update network weights
        optimizer_weights.zero_grad()
        logits_weights, aux_logits_weights = self(input_train)
        loss_weights = F.cross_entropy(logits_weights, target_train)
        if aux_logits_weights is not None:
            aux_loss_weights = F.cross_entropy(aux_logits_weights, target_train)
            loss_weights += self.config["model"]["auxiliary_weight"] * aux_loss_weights
        self.manual_backward(loss_weights)
        optimizer_weights.step()

        self.log("train_loss_weights", loss_weights)
        self.log("train_loss_arch", loss_arch)

        return {"loss": loss_weights}

    def on_train_epoch_start(self):
        self.search_space.update_drop_path_prob(
            self.drop_path_scheduler(self.current_epoch)
        )
        self.search_space.update_temperature(
            self.temperature_scheduler(self.current_epoch)
        )
        self.log("drop_path_prob", self.search_space.drop_path_prob)
        self.log("temperature", self.search_space.temperature)
