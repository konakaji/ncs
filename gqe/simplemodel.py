import numpy as np

from gqe.estimator import EnergyEstimator
from qwrapper.obs import PauliObservable
from qwrapper.optimizer import AdamOptimizer
from qwrapper.operator import PauliTimeEvolution, ControllablePauli
from qwrapper.sampler import DefaultImportantSampler
from gqe.util import identity
from gqe.estimator import Sampler


class Ansatz:
    def __init__(self, h_vec, o_vec: [PauliObservable], nqubit):
        self.h_vec = h_vec
        self._o_vec = o_vec
        self.nqubit = nqubit

    def get_positive_h_vec(self):
        return [abs(h) for h in self.h_vec]

    def get_signed_o_vec(self):
        rs = []
        for j, h in enumerate(self.h_vec):
            if h < 0:
                rs.append(ControllablePauli(self._o_vec[j].p_string, -1 * self._o_vec[j].sign))
            else:
                rs.append(ControllablePauli(self._o_vec[j].p_string, self._o_vec[j].sign))
        return rs

    def copy(self):
        return Ansatz(self.h_vec, self._o_vec, self.nqubit)


class SimpleSampler(Sampler):
    def __init__(self, ansatz: Ansatz, N, lam):
        hs = ansatz.get_positive_h_vec()
        hs.append(lam - sum(hs))
        self.hs = hs
        operators = ansatz.get_signed_o_vec()
        operators.append(identity(ansatz.nqubit))
        self.operators = operators
        self.evolutions = [PauliTimeEvolution(o, lam / N) for o in operators]
        self.sampler = DefaultImportantSampler(self.hs)

    def sample_operators(self, count=1) -> [ControllablePauli]:
        res = []
        for j in range(count):
            index = self.sampler.sample_index()
            res.append(self.operators[index])
        return res

    def sample_time_evolutions(self, count=1) -> [PauliTimeEvolution]:
        res = []
        for j in range(count):
            index = self.sampler.sample_index()
            res.append(self.evolutions[index])
        return res

    @DeprecationWarning
    def sample(self, count=1):
        return self.sample_operators(count)

    def get(self, j):
        return self.operators[j]


class SimpleModel:
    def __init__(self, estimator: EnergyEstimator,
                 ansatz: Ansatz, N, lam, n_sample, tool="qulacs"):
        self.estimator = estimator
        self.ansatz = ansatz
        self.N = N
        self.lam = lam
        self.n_sample = n_sample
        self.tool = tool

    def run(self, optimizer: AdamOptimizer):
        optimizer.do_optimize(self.gradient, self.ansatz.h_vec, self.cost)

    def cost(self, params):
        results = []
        for _ in range(self.n_sample):
            ansatz = self.ansatz.copy()
            ansatz.h_vec = params
            sampler = SimpleSampler(self.ansatz, self.N, self.lam)
            results.append(self.estimator.value(sampler))
        return np.mean(results), np.std(results), np.sum(self.ansatz.get_positive_h_vec())

    def gradient(self, params):
        self.ansatz.h_vec = params
        if sum(self.ansatz.get_positive_h_vec()) > self.lam:
            raise AttributeError('sum of h became larger than h_vec')
        sampler = SimpleSampler(self.ansatz, self.N, self.lam)
        res = np.array([self.estimator.grad(sampler, j) for j in range(len(params))])
        results = []
        for h, r in zip(self.ansatz.h_vec, res):
            if h < 0:
                results.append(-r)
            else:
                results.append(r)
        return np.array(results)
