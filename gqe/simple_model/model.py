import numpy as np

from qwrapper.obs import PauliObservable
from qwrapper.optimizer import AdamOptimizer
from qwrapper.operator import PauliTimeEvolution, ControllablePauli
from qwrapper.sampler import FasterImportantSampler
from qswift.compiler import DefaultOperatorPool
from gqe.energy_estimator.qswift import SecondQSwiftEstimator
from gqe.util import identity
from gqe.energy_estimator.ee import Sampler, EnergyEstimator


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
    def __init__(self):
        self.cache = {}
        self.hs = None
        self.operators = None
        self.evolutions = None
        self.sampler = None

    def reset(self, ansatz: Ansatz, N, lam):
        hs = ansatz.get_positive_h_vec()
        hs.append(lam - sum(hs))
        self.hs = hs
        operators = ansatz.get_signed_o_vec()
        operators.append(identity(ansatz.nqubit))
        self.operators = operators
        evolutions = []
        for o in operators:
            key = str(o)
            if key not in self.cache:
                evolution = PauliTimeEvolution(o, lam / N)
                self.cache[key] = evolution
            evolutions.append(self.cache[key])
        self.evolutions = evolutions
        self.sampler = FasterImportantSampler(self.hs)

    def sample_indices(self, count=1):
        res = []
        for j in range(count):
            index = self.sampler.sample_index()
            res.append(index)
        return res

    def sample_operators(self, count=1) -> [ControllablePauli]:
        res = []
        for j in range(count):
            index = self.sampler.sample_index()
            res.append(self.operators[index])
        return res

    def sample_time_evolutions(self, count=1) -> [PauliTimeEvolution]:
        res = []
        for index in self.sampler.sample_indices(count):
            res.append(self.evolutions[index])
        return res

    def get(self, j):
        return self.operators[j]


class SimpleModel:
    def __init__(self, estimator: SecondQSwiftEstimator,
                 ansatz: Ansatz, N, lam, tool="qulacs"):
        self.estimator = estimator
        self.ansatz = ansatz
        self.N = N
        self.lam = lam
        self.tool = tool

    def run(self, optimizer: AdamOptimizer):
        optimizer.do_optimize(self.gradient, self.ansatz.h_vec, self.cost)

    def cost(self, params):
        ansatz = self.ansatz.copy()
        ansatz.h_vec = params
        sampler = FasterImportantSampler(ansatz.get_positive_h_vec())
        pool = DefaultOperatorPool(ansatz.get_signed_o_vec())
        return self.estimator.value(sampler, pool, self.lam)

    def gradient(self, params):
        self.ansatz.h_vec = params
        summation = sum(self.ansatz.get_positive_h_vec())
        if summation > self.lam:
            raise AttributeError(f'sum of h became larger than h_vec: {summation}')
        sampler = FasterImportantSampler(self.ansatz.get_positive_h_vec())
        pool = DefaultOperatorPool(self.ansatz.get_signed_o_vec())
        res = np.array([self.estimator.first_order_grad(sampler, pool, self.lam, j) for j in
                        range(len(params))])
        results = []
        for h, r in zip(self.ansatz.h_vec, res):
            if h < 0:
                results.append(-r)
            else:
                results.append(r)
        return np.array(results)
