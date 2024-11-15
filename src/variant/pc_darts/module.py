import torch
import torch.nn.functional as F

from base.darts_model import BaseDARTSModel
from component.schedulers import DropPathScheduler, TemperatureScheduler
from variant.pc_darts.search_space import PCDARTSSearchSpace


class PCDARTSModule(BaseDARTSModel):
    def __init__(self, config: dict):
        search_space = PCDARTSSearchSpace(
            num_nodes=config["model"]["num_nodes"],
            in_channels=config["model"]["in_channels"],
            num_partial_channel_connections=config["model"][
                "num_partial_channel_connections"
            ],
            edge_norm_init=config["model"].get("edge_norm_init", 1.0),
            edge_norm_strength=config["model"].get("edge_norm_strength", 1.0),
            num_segments=config["model"].get("num_segments", 4),
            temperature_start=config["model"].get("temperature_start", 1.0),
            drop_path_prob_start=config["training"].get("drop_path_prob_start", 0.0),
        )
        super().__init__(
            search_space=search_space,
            config=config,
            features={"auxiliary_head": True},
        )

        self.edge_norm_params = list(self.search_space.edge_norms.parameters())

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
        optimizer_weights, optimizer_arch, optimizer_edge_norm = self.optimizers()  # type: ignore

        # Unpack the batch
        input_train, target_train = batch["train"]
        input_search, target_search = batch["search"]

        # Debug shapes
        print(f"Input search shape: {input_search.shape}")

        # Update architecture parameters
        optimizer_arch.zero_grad()
        try:
            logits_arch, aux_logits_arch = self(input_search)
            print(
                f"Logits shape: {logits_arch.shape}, Aux logits shape: {aux_logits_arch.shape}"
            )
        except Exception as e:
            print(f"Error in forward pass: {e}")
            print(f"Input shape: {input_search.shape}")
            raise e

        loss_arch = F.cross_entropy(logits_arch, target_search)
        if aux_logits_arch is not None:
            aux_loss_arch = F.cross_entropy(aux_logits_arch, target_search)
            loss_arch += self.config["model"]["auxiliary_weight"] * aux_loss_arch
        self.manual_backward(loss_arch)
        optimizer_arch.step()

        # Update edge normalization parameters
        optimizer_edge_norm.zero_grad()
        logits_edge_norm, aux_logits_edge_norm = self(input_search)
        loss_edge_norm = F.cross_entropy(logits_edge_norm, target_search)
        if aux_logits_edge_norm is not None:
            aux_loss_edge_norm = F.cross_entropy(aux_logits_edge_norm, target_search)
            loss_edge_norm += (
                self.config["model"]["auxiliary_weight"] * aux_loss_edge_norm
            )
        self.manual_backward(loss_edge_norm)
        optimizer_edge_norm.step()

        # Update network weights
        optimizer_weights.zero_grad()
        logits_weights, aux_logits_weights = self(input_train)
        loss_weights = F.cross_entropy(logits_weights, target_train)
        if aux_logits_weights is not None:
            aux_loss_weights = F.cross_entropy(aux_logits_weights, target_train)
            loss_weights += self.config["model"]["auxiliary_weight"] * aux_loss_weights
        self.manual_backward(loss_weights)
        optimizer_weights.step()

        # Log metrics
        self.log("train_loss_weights", loss_weights)
        self.log("train_loss_arch", loss_arch)
        self.log("train_loss_edge_norm", loss_edge_norm)

        return {"loss": loss_weights}

    def on_train_epoch_start(self):
        # Update schedulers
        self.search_space.update_drop_path_prob(
            self.drop_path_scheduler(self.current_epoch)
        )
        self.search_space.update_temperature(
            self.temperature_scheduler(self.current_epoch)
        )

        # Log values
        self.log("drop_path_prob", self.search_space.drop_path_prob)
        self.log("temperature", self.search_space.temperature)

    def configure_optimizers(self):
        return super().configure_optimizers()

    def test_step(self, batch, batch_idx: int) -> dict[str, float | torch.Tensor]:
        """Implement test step for PC-DARTS."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()

        # Log metrics
        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return {"test_loss": loss, "test_acc": acc}
