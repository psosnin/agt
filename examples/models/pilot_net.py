"""PilotNet architecture from https://arxiv.org/abs/1604.07316"""

import torch
from collections import OrderedDict


class PilotNet(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(3, 24, 5, 2, 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(24, 36, 5, 2, 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(36, 48, 5, 2, 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 64, 3, 1, 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1, 0),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(1152, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )

    def __getitem__(self, idx: slice | int) -> torch.nn.Module | torch.nn.Sequential:
        if isinstance(idx, slice):
            return torch.nn.Sequential(OrderedDict(list(self._modules.items())[idx]))
        return self._get_item_by_idx(self._modules.values(), idx)
