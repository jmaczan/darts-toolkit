from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD


def get_default_weights_optimizer(weight_params, config) -> Optimizer:
    return SGD(
        params=weight_params,
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"],
    )


def get_default_arch_optimizer(arch_params, config) -> Optimizer:
    return Adam(
        params=arch_params,
        lr=config["training"]["arch_learning_rate"],
        betas=(0.5, 0.999),
        weight_decay=config["training"]["arch_weight_decay"],
    )


def get_default_edge_norm_optimizer(edge_norm_params, config) -> Optimizer:
    return Adam(
        params=edge_norm_params,
        lr=config["training"]["edge_norm_learning_rate"],
        betas=(0.5, 0.999),
        weight_decay=config["training"]["edge_norm_weight_decay"],
    )
