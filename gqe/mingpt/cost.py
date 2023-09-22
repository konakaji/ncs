from abc import abstractmethod, ABC
from gqe.operator_pool.op import ListablePool
from qswift.sequence import Sequence
from qswift.initializer import CircuitInitializer
from qwrapper.obs import Hamiltonian
from gqe.util import get_device
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
                 pool: ListablePool, taus, nshot=0, tool='qulacs', device=None):
        if device is None:
            device = get_device()
        self.sequence = Sequence(obs, initializer, pool, taus=taus, nshot=nshot, tool=tool)
        self.device = device

    def energy(self, idx):
        """
        :param idx: shape(# of data, length of sequence)
        :return energy
        """
        energies = []
        for seq in idx:
            energies.append(self.sequence.evaluate(seq.detach().cpu().numpy()))
        return torch.tensor(energies, dtype=torch.float).to(self.device)

    def vocab_size(self):
        return self.sequence.operator_pool.size() * len(self.sequence.taus)
