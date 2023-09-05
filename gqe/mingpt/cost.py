from abc import abstractmethod, ABC
from gqe.operator_pool.op import ListablePool
from qswift.sequence import Sequence
from qswift.initializer import CircuitInitializer
from qwrapper.obs import Hamiltonian
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


class EnergyCost(Cost):
    def __init__(self, obs: Hamiltonian,
                 initializer: CircuitInitializer,
                 pool: ListablePool, taus, nshot=0, tool='qulacs'):
        self.sequence = Sequence(obs, initializer, pool, taus=taus, nshot=nshot, tool=tool)

    def energy(self, idx):
        indices = idx[0][1:].detach().numpy()
        return torch.tensor(self.sequence.evaluate(indices))

    def vocab_size(self):
        return self.sequence.operator_pool.size() * len(self.sequence.taus)
