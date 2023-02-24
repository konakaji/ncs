import random, numpy as np, sys

from qwrapper.operator import PauliTimeEvolution
from qwrapper.obs import PauliObservable
from qwrapper.measurement import MeasurementMethod
from qwrapper.circuit import init_circuit
from qwrapper.optimizer import AdamOptimizer
from gqe.ansatz import Ansatz
from gqe.sampler import OperatorSampler


class GQE:
    def __init__(self, ansatz: Ansatz, mes_method: MeasurementMethod, N, tau, n_sample, ntotal=0, tool="qulacs"):
        self.ansatz = ansatz
        self.mes_method = mes_method
        self.N = N
        self.tau = tau
        self.n_sample = n_sample
        self.nqubit = mes_method.hamiltonian.nqubit
        self.tool = tool
        self.ntotal = ntotal
        self._targets = [j for j in range(self.nqubit)]

    def run(self, optimizer: AdamOptimizer):
        optimizer.do_optimize(self.gradient, self.ansatz.h_vec, self.cost)

    def cost(self, params):
        results = []
        for _ in range(self.n_sample):
            results.append(self._cost(params))
        return np.mean(results), np.std(results), np.sum(self.ansatz.get_positive_h_vec())

    def gradient(self, params):
        self.ansatz.h_vec = params
        if sum(self.ansatz.get_positive_h_vec()) > self._lam():
            raise AttributeError('sum of h became larger than h_vec')
        sampler = OperatorSampler(self.ansatz, self._lam(), self.nqubit)
        res = np.array([self.prob_gradient(j, sampler, self.tau) for j in range(len(params))])
        results = []
        for h, r in zip(self.ansatz.h_vec, res):
            if h < 0:
                results.append(-r)
            else:
                results.append(r)
        return np.array(results)

    def prob_gradient(self, j, sampler, tau):
        res = 0
        for _ in range(self.n_sample):
            res += self._prob_gradient(j, sampler, tau) / self.n_sample
        return res

    def _cost(self, params):
        ansatz = self.ansatz.copy()
        ansatz.h_vec = params

        def prepare():
            sampler = OperatorSampler(self.ansatz, self._lam(), self.nqubit)
            operators = self._to_time_evolution(sampler.sample(self.N), self.tau)
            qc = init_circuit(self.nqubit, self.tool)
            for o in operators:
                o.add_circuit(qc)
            return qc

        return self.mes_method.get_value(prepare, ntotal=self.ntotal)

    def _prob_gradient(self, j, sampler, tau):
        seed = random.randint(0, sys.maxsize)

        def prepare_first():
            l = random.randint(1, self.N)
            f = sampler.sample(l - 1)
            former = self._to_time_evolution(f, tau)
            latter = self._to_time_evolution(sampler.sample(self.N - l), tau)
            j_evolution = self._to_time_evolution([sampler.get(j)], tau)[0]
            qc = init_circuit(self.nqubit, self.tool)
            for o in former:
                o.add_circuit(qc)
            j_evolution.add_circuit(qc)
            for o in latter:
                o.add_circuit(qc)
            return qc

        def prepare_second():
            l = random.randint(1, self.N)
            f = sampler.sample(l - 1)
            former = self._to_time_evolution(f, tau)
            latter = self._to_time_evolution(sampler.sample(self.N - l), tau)
            qc = init_circuit(self.nqubit, self.tool)
            for o in former:
                o.add_circuit(qc)
            for o in latter:
                o.add_circuit(qc)
            return qc

        val = (self.mes_method.get_value(prepare_first, self.ntotal, seed=seed)
               - self.mes_method.get_value(prepare_second, self.ntotal, seed=seed)) / tau
        return val

    def _lam(self):
        return self.N * self.tau

    @classmethod
    def _to_time_evolution(cls, paulis: [PauliObservable], tau) -> [PauliTimeEvolution]:
        return [PauliTimeEvolution(p, tau) for p in paulis]
