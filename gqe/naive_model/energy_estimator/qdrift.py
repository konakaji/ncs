import numpy as np
from qwrapper.circuit import init_circuit, QWrapper
from qwrapper.measurement import MeasurementMethod
from qwrapper.obs import Hamiltonian, PauliObservable
from qwrapper.operator import ControllablePauli, PauliTimeEvolution
from qwrapper.sampler import FasterImportantSampler
import abc
from abc import abstractmethod

import random, sys

"""
This package is deprecated. Please use gqe.iid.IIDEstimator instead.
"""


class Initializer:
    def initialize(self, qc, targets) -> QWrapper:
        return qc


class XInitializer(Initializer):
    def initialize(self, qc, targets) -> QWrapper:
        for t in targets:
            qc.h(t)
        return qc


class AncillaMeasurementMethod(MeasurementMethod):
    def __init__(self, hamiltonian: Hamiltonian):
        paulis = []
        for p in hamiltonian.paulis:
            paulis.append(PauliObservable(p.p_string + "X", p.sign))
        super().__init__(Hamiltonian(hamiltonian.hs, paulis, hamiltonian.nqubit))


class StochasticMeasurementMethod(MeasurementMethod):
    def __init__(self, hamiltonian: Hamiltonian, n_samples):
        self.n_samples = n_samples
        hs = []
        signs = []
        for h in hamiltonian.hs:
            hs.append(abs(h))
            sign = 1 if h > 0 else -1
            signs.append(sign)
        self.sampler = FasterImportantSampler(hs)
        self.signs = signs
        super().__init__(hamiltonian)

    def exact_values(self, prepare):
        res = []
        paulis = self.hamiltonian.paulis
        for index in self.sampler.sample_indices(self.n_samples):
            qc = prepare()
            pauli = paulis[index]
            sign = self.signs[index]
            res.append(sign * pauli.exact_value(qc))
        return res

    def get_values(self, prepare, ntotal=0, seed=None):
        if seed is not None:
            random.seed(seed)
        if ntotal == 0:
            return self.exact_values(prepare)
        res = []
        paulis = self.hamiltonian.paulis
        nshot = int(ntotal / self.n_samples)
        for index in self.sampler.sample_indices(self.n_samples):
            qc = prepare()
            pauli = paulis[index]
            sign = self.signs[index]
            res.append(sign * pauli.get_value(qc, nshot))
        return res


class AncillaStochasticMeasurementMethod(StochasticMeasurementMethod):
    def __init__(self, hamiltonian: Hamiltonian, n_samples):
        paulis = []
        for p in hamiltonian.paulis:
            paulis.append(PauliObservable(p.p_string + "X", p.sign))
        super().__init__(Hamiltonian(hamiltonian.hs, paulis, hamiltonian.nqubit), n_samples)


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


class QDriftEstimator:
    def __init__(self, obs: Hamiltonian, N, measurement=None, ancilla_measurement=None, tool='qulacs', shot=0):
        self.nqubit = obs.nqubit
        if measurement is None:
            self.mes_method = MeasurementMethod(obs)
        else:
            self.mes_method = measurement
        if ancilla_measurement is None:
            self.ancilla_mes_method = AncillaMeasurementMethod(obs)
        else:
            self.ancilla_mes_method = ancilla_measurement
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
        seed = random.randint(0, sys.maxsize)
        get_operator_func = self._get_operator(sampler, index)
        print("--------")
        t_1 = self.ancilla_mes_method.get_value(self._get_prepare(sampler, get_operator_func, False),
                                                ntotal=self.shot,
                                                seed=seed)
        t_2 = self.ancilla_mes_method.get_value(
            self._get_prepare(sampler, get_operator_func, True), ntotal=self.shot, seed=seed)
        return (t_1 + t_2) / 2

    def grads(self, sampler: Sampler):
        indices = []
        seed = random.randint(0, sys.maxsize)

        def get_operator():
            index = sampler.sample_indices(1)[0]
            indices.append(index)
            return sampler.get(index)

        def get_operator_inv():
            index = sampler.sample_indices(1)[0]
            return sampler.get(index)

        values = (np.array(
            self.ancilla_mes_method.get_values(self._get_prepare(sampler, get_operator, False), ntotal=self.shot,
                                               seed=seed)) -
                  np.array(self.ancilla_mes_method.get_values(self._get_prepare(sampler, get_operator_inv, True),
                                                              ntotal=self.shot,
                                                              seed=seed))) / 2
        return values, indices

    def _get_prepare(self, sampler, get_operator, inverse):
        def prepare():
            pos = random.randint(0, self.N - 1)
            qc = self.initializer.initialize(init_circuit(self.nqubit + 1, tool=self.tool),
                                             targets=self._targets)
            qc.h(self._ancilla)
            evolutions = sampler.sample_time_evolutions(self.N)
            for j in range(self.N):
                if j == pos:
                    operator = get_operator()
                    self._add_swift_operator(qc, operator, inverse)
                    continue
                evolutions[j].add_circuit(qc)
            return qc

        return prepare

    def _get_operator(self, sampler, index):
        def get_operator():
            return sampler.get(index)

        return get_operator

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
