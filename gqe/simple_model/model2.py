import json, numpy as np

from qwrapper.obs import PauliObservable, Hamiltonian
from qwrapper.optimizer import AdamOptimizer
from qwrapper.operator import ControllablePauli
from qwrapper.sampler import FasterImportantSampler
from qswift.compiler import DefaultOperatorPool
from gqe.energy_estimator.iid import IIDEstimator
from gqe.util import identity


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

    def toJSON(self):
        map = {
            'h_vec': [str(h) for h in self.h_vec],
            'o_vec': [str(o) for o in self._o_vec],
            'nqubit': self.nqubit
        }
        return json.dumps(map, indent=2)

    @classmethod
    def fromJSON(cls, string):
        map = json.loads(string)
        h_vec = [float(h) for h in map['h_vec']]
        o_vec = []
        for s in map['o_vec']:
            sign_str = s[0]
            if sign_str == "+":
                sign = 1
            else:
                sign = -1
            o_vec.append(PauliObservable(s[1:], sign))
        nqubit = map['nqubit']
        return Ansatz(h_vec, o_vec, nqubit)


class SimpleModel:
    def __init__(self, estimator: IIDEstimator,
                 ansatz: Ansatz, N, lam, tool="qulacs", exact_cost=True):
        self.estimator = estimator
        self.ansatz = ansatz
        self.N = N
        self.lam = lam
        self.tool = tool
        self.exact_cost = exact_cost

    def run(self, optimizer: AdamOptimizer):
        optimizer.do_optimize(self.gradient, self.ansatz.h_vec, self.cost)

    def cost(self, params):
        ansatz = self.ansatz.copy()
        ansatz.h_vec = params
        if self.exact_cost:
            time_evolution = Hamiltonian(ansatz.get_positive_h_vec(), ansatz.get_signed_o_vec(), self.ansatz.nqubit)
            return self.estimator.exact(time_evolution)
        hs = ansatz.get_positive_h_vec()
        hs.append(self.lam - sum(hs))
        sampler = FasterImportantSampler(hs)
        operators = ansatz.get_signed_o_vec()
        operators.append(identity(self.ansatz.nqubit))
        pool = DefaultOperatorPool(operators)
        return self.estimator.value(sampler, pool, self.lam)

    def gradient(self, params):
        self.ansatz.h_vec = params
        summation = sum(self.ansatz.get_positive_h_vec())
        if summation > self.lam:
            raise AttributeError(f'sum of h became larger than h_vec: {summation}')
        hs = self.ansatz.get_positive_h_vec()
        hs.append(self.lam - summation)
        sampler = FasterImportantSampler(hs)
        operators = self.ansatz.get_signed_o_vec()
        operators.append(identity(self.ansatz.nqubit))
        pool = DefaultOperatorPool(operators)

        grads = []
        for j in range(len(params)):
            sign = 1 if self.ansatz.h_vec[j] > 0 else -1
            grads.append(sign * self.estimator.grad(sampler, pool, self.lam, j))
        return np.array(grads)
