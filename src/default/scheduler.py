from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler


def get_default_weights_scheduler(optimizer, config) -> LRScheduler:
    return CosineAnnealingLR(
        optimizer=optimizer, T_max=config["training"]["max_epochs"]
    )
