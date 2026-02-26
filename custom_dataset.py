import torch
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Tuple


class CustomDataset(Dataset):
    def __init__(self, inputs, targets) -> None:
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index) -> Tuple:
        return self.inputs[index], self.targets[index]
