from abc import abstractmethod, ABC
import torch


class Cost(ABC):
    @abstractmethod
    def energy(self, idx):
        return 0


class IndicesCost(Cost):
    def __init__(self, indices):
        self.indices = indices

    def energy(self, idx):
        result = idx[0][1:]
        bool_tensor = (self.indices - result) == 0
        size = result.size(0)
        diff = torch.sum(bool_tensor).item()
        return (size - diff) / size
