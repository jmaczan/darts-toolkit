import torch.nn as nn


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
