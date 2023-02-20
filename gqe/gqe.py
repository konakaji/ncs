import random, numpy as np

from qwrapper.operator import PauliTimeEvolution
from qwrapper.obs import PauliObservable
from qwrapper.measurement import MeasurementMethod
from qwrapper.circuit import QWrapper
from qwrapper.circuit import init_circuit
from qwrapper.optimizer import AdamOptimizer
from gqe.ansatz import Ansatz
from gqe.measurement import AncillaMeasurementMethod
from gqe.sampler import OperatorSampler


class GQE:
    def __init__(self, ansatz: Ansatz, mes_method: MeasurementMethod,
                 N, n_sample, ntotal=0, tool="qulacs"):
        self.ansatz = ansatz
        self.mes_method = mes_method
        self.ancilla_mes_method = AncillaMeasurementMethod(self.mes_method.hamiltonian)
        self.N = N
        self.n_sample = n_sample
        self.nqubit = mes_method.hamiltonian.nqubit
        self.tool = tool
        self.ntotal = ntotal
        self._targets = [j for j in range(self.nqubit)]

    def run(self, optimizer: AdamOptimizer):
        optimizer.do_optimize(self.gradient, self.ansatz.h_vec, self.cost)

    def cost(self, params):
        N = 3000
        ansatz = self.ansatz.copy()
        ansatz.h_vec = params
        sampler = OperatorSampler(self.ansatz)
        tau = self.ansatz.lam() / N
        operators = self._to_time_evolution(sampler.sample(N), tau)

        def prepare():
            qc = init_circuit(self.nqubit, self.tool)
            for o in operators:
                o.add_circuit(qc)
            return qc

        return self.mes_method.get_value(prepare, ntotal=self.ntotal)

    def gradient(self, params):
        self.ansatz.h_vec = params
        sampler = OperatorSampler(self.ansatz)
        tau = self.ansatz.lam() / self.N
        offset = self.operator_gradient(sampler, tau)
        res = np.array([self.prob_gradient(j, sampler, tau) + offset for j in range(len(params))])
        results = []
        for h, r in zip(self.ansatz.h_vec, res):
            if h < 0:
                results.append(-r)
            else:
                results.append(r)
        return np.array(results)

    def operator_gradient(self, sampler, tau):
        res = 0
        for _ in range(self.n_sample):
            res += self._operator_gradient(sampler, tau) / self.n_sample
        return res

    def prob_gradient(self, j, sampler, tau):
        res = 0
        for _ in range(self.n_sample):
            res += self._prob_gradient(j, sampler, tau) / self.n_sample
        return res

    def _prob_gradient(self, j, sampler, tau):
        l = random.randint(1, self.N)
        former = self._to_time_evolution(sampler.sample(l - 1), tau)
        latter = self._to_time_evolution(sampler.sample(self.N - l), tau)
        j_evolution = self._to_time_evolution([sampler.get(j)], tau)[0]
        one_evolution = self._to_time_evolution(sampler.sample(1), tau)[0]

        def prepare_first():
            qc = init_circuit(self.nqubit, self.tool)
            for o in former:
                o.add_circuit(qc)
            j_evolution.add_circuit(qc)
            for o in latter:
                o.add_circuit(qc)
            qc.draw_and_show()
            return qc

        def prepare_second():
            qc = init_circuit(self.nqubit, self.tool)
            for o in former:
                o.add_circuit(qc)
            one_evolution.add_circuit(qc)
            for o in latter:
                o.add_circuit(qc)
            return qc

        val = (self.mes_method.get_value(prepare_first, self.ntotal)
               - self.mes_method.get_value(prepare_second, self.ntotal)) / tau
        return val

    def _operator_gradient(self, sampler, tau):
        l = random.randint(1, self.N)
        ancilla = self.nqubit
        former = self._to_time_evolution(sampler.sample(l), tau)
        pauli = sampler.sample(1)[0]
        latter = self._to_time_evolution(sampler.sample(self.N - l), tau)

        def prepare_first():
            qc = init_circuit(self.nqubit + 1, self.tool)
            qc.h(ancilla)
            for o in former:
                o.add_circuit(qc)
            self._add_swift_gate(qc, pauli, 0, ancilla)
            for o in latter:
                o.add_circuit(qc)
            return qc

        def prepare_second():
            qc = init_circuit(self.nqubit + 1, self.tool)
            qc.h(ancilla)
            for o in former:
                o.add_circuit(qc)
            self._add_swift_gate(qc, pauli, 1, ancilla)
            for o in latter:
                o.add_circuit(qc)
            return qc

        return self.ancilla_mes_method.get_value(prepare_first, self.ntotal) \
               + self.ancilla_mes_method.get_value(prepare_second, self.ntotal)

    def _add_swift_gate(self, qc: QWrapper, pauli, b, ancilla):
        qc.s(ancilla)
        if pauli.sign == -1:
            qc.z(ancilla)
        if b == 0:
            qc.z(ancilla)
            qc.x(ancilla)
            pauli.add_controlled_circuit(ancilla, self._targets, qc)
            qc.x(ancilla)
        else:
            pauli.add_controlled_circuit(ancilla, self._targets, qc)

    @classmethod
    def _to_time_evolution(cls, paulis: [PauliObservable], tau) -> [PauliTimeEvolution]:
        return [PauliTimeEvolution(p, tau) for p in paulis]
