from qwrapper.measurement import MeasurementMethod
from qwrapper.obs import Hamiltonian, PauliObservable
from qwrapper.sampler import FasterImportantSampler
import random


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
