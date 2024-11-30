import torch
import torch.nn as nn
import torch.nn.functional as F


class Node(nn.Module):
    def __init__(self):
        super().__init__()
        self.edges = nn.ModuleList()


class Edge(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixed_operation = MixedOperation()
        self.alpha = nn.Parameter(torch.randn(self.mixed_operation.ops.size()))


class MixedOperation(nn.Module):
    def __init__(self):
        super().__init__()
        self.ops = nn.ModuleList()


class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        self.nodes = nn.ModuleList()


def mini_darts():
    pass


if __name__ == "__main__":
    mini_darts()
