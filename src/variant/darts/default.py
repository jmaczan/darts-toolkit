from base.operations import (
    AvgPool,
    DynamicSizeConv2d,
    DynamicSizeSeparableConv2d,
    Identity,
    MaxPool,
    ZeroOp,
)


def default_darts_candidate_operations():
    return [
        Identity(),
        DynamicSizeConv2d(kernel_size=3, padding=1),
        DynamicSizeConv2d(kernel_size=5, padding=2),
        MaxPool(kernel_size=3, stride=1, padding=1),
        AvgPool(kernel_size=3, stride=1, padding=1),
        DynamicSizeSeparableConv2d(kernel_size=3, padding=1),
        DynamicSizeSeparableConv2d(kernel_size=5, padding=2),
        ZeroOp(),
    ]
