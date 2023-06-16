import abc
from abc import abstractmethod

from qwrapper.obs import Hamiltonian
from qwrapper.operator import ControllablePauli, PauliTimeEvolution


class Sampler(abc.ABC):
    @abstractmethod
    def sample_indices(self, count=1):
        pass

    @abstractmethod
    def sample_operators(self, count=1) -> [ControllablePauli]:
        pass

    @abstractmethod
    def sample_time_evolutions(self, count=1) -> [PauliTimeEvolution]:
        pass

    @abstractmethod
    def get(self, index) -> ControllablePauli:
        pass


class EnergyEstimator(abc.ABC):
    @abstractmethod
    def value(self, indices):
        pass


class GEnergyEstimator(abc.ABC):
    def __init__(self, obs: Hamiltonian):
        self.hamiltonian = obs

    @abstractmethod
    def value(self, sampler: Sampler):
        pass

    @abstractmethod
    def grad(self, sampler: Sampler, index):
        pass

    @abstractmethod
    def grads(self, sampler: Sampler):
        pass
