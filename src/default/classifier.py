import torch.nn as nn


def get_default_classifier(in_features, out_features):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(),
        nn.Linear(
            in_features=in_features,
            out_features=out_features,
        ),
    )
