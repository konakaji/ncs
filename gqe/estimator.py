import abc
from abc import abstractmethod

from qwrapper.circuit import init_circuit, QWrapper
from qwrapper.obs import Hamiltonian
from qwrapper.operator import PauliTimeEvolution, ControllablePauli
from gqe.measurement import MeasurementMethod, AncillaMeasurementMethod
import random, sys


class Initializer:
    def initialize(self, qc, targets) -> QWrapper:
        return qc


class XInitializer(Initializer):
    def initialize(self, qc, targets) -> QWrapper:
        for t in targets:
            qc.h(t)
        return qc


class Sampler(abc.ABC):
    @abstractmethod
    def sample_operators(self, count=1) -> [ControllablePauli]:
        pass

    @abstractmethod
    def sample_time_evolutions(self, count=1) -> [PauliTimeEvolution]:
        pass

    @abstractmethod
    def get(self, index) -> ControllablePauli:
        pass


class EnergyEstimator:
    def __init__(self, hamiltonian: Hamiltonian):
        self.hamiltonian = hamiltonian

    def value(self, sampler: Sampler):
        pass

    def grad(self, sampler: Sampler, index):
        pass


class QDriftEstimator(EnergyEstimator):
    def __init__(self, hamiltonian: Hamiltonian, N, tool='qulacs', shot=0):
        super().__init__(hamiltonian)
        self.nqubit = hamiltonian.nqubit
        self.mes_method = MeasurementMethod(hamiltonian)
        self.ancilla_mes_method = AncillaMeasurementMethod(hamiltonian)
        self.N = N
        self.initializer = XInitializer()
        self._ancilla = self.nqubit
        self._targets = [j for j in range(self.nqubit)]
        self.tool = tool
        self.shot = shot

    def value(self, sampler: Sampler):
        def prepare():
            qc = self.initializer.initialize(init_circuit(self.nqubit, tool=self.tool), self._targets)
            for o in sampler.sample_time_evolutions(self.N):
                o.add_circuit(qc)
            return qc

        return self.mes_method.get_value(prepare, ntotal=self.shot)

    def grad(self, sampler: Sampler, index):
        def prepare():
            pos = random.randint(0, self.N - 1)
            qc = self.initializer.initialize(init_circuit(self.nqubit + 1, tool=self.tool),
                                             targets=self._targets)
            qc.h(self._ancilla)
            evolutions = sampler.sample_time_evolutions(self.N)
            for j in range(self.N):
                if j == pos:
                    self._add_swift_operator(qc, sampler.get(index))
                    continue
                evolutions[j].add_circuit(qc)
            return qc

        def prepare_inverse():
            pos = random.randint(0, self.N - 1)
            qc = self.initializer.initialize(init_circuit(self.nqubit + 1, tool=self.tool),
                                             targets=self._targets)
            qc.h(self._ancilla)
            evolutions = sampler.sample_time_evolutions(self.N)
            for j in range(self.N):
                if j == pos:
                    self._add_swift_operator(qc, sampler.get(index), True)
                    continue
                evolutions[j].add_circuit(qc)
            return qc

        seed = random.randint(0, sys.maxsize)
        return (self.ancilla_mes_method.get_value(prepare, ntotal=self.shot,
                                                  seed=seed) + self.ancilla_mes_method.get_value(prepare_inverse,
                                                                                                 ntotal=self.shot,
                                                                                                 seed=seed)) / 2

    def _add_swift_operator(self, qc, operator, inverse=False):
        qc.s(self._ancilla)
        pauli = operator
        if pauli.sign == -1:
            qc.z(self._ancilla)
        if inverse:
            qc.z(self._ancilla)
            qc.x(self._ancilla)
            pauli.add_controlled_circuit(self._ancilla, self._targets, qc)
            qc.x(self._ancilla)
        else:
            pauli.add_controlled_circuit(self._ancilla, self._targets, qc)
