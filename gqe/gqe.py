import random

from qwrapper.operator import PauliTimeEvolution
from qwrapper.obs import PauliObservable
from qwrapper.measurement import MeasurementMethod
from qwrapper.circuit import QWrapper
from qwrapper.circuit import init_circuit
from gqe.ansatz import Ansatz
from gqe.measurement import AncillaMeasurementMethod
from gqe.sampler import OperatorSampler


class GQE:
    def __init__(self, mes_method: MeasurementMethod, nqubit,
                 optimizer, N, n_sample, ntotal=0, tool="qulacs"):
        self.mes_method = mes_method
        self.ancilla_mes_method = AncillaMeasurementMethod(self.mes_method.hamiltonian)
        self.optimizer = optimizer
        self.N = N
        self.n_sample = n_sample
        self.nqubit = nqubit
        self.tool = tool
        self.ntotal = ntotal
        self._targets = [j for j in range(nqubit)]

    def step(self, ansatz: Ansatz):
        sampler = OperatorSampler(ansatz)
        tau = ansatz.lam() / self.N
        offset = self.operator_gradient(sampler, tau)
        grad_vec = [self.prob_gradient(j, sampler, tau) + offset for j in range(self.N)]

    def operator_gradient(self, sampler, tau):
        res = 0
        for _ in self.n_sample:
            res += self._operator_gradient(sampler, tau)
        return res

    def prob_gradient(self, j, sampler, tau):
        res = 0
        for _ in self.n_sample:
            res += self._prob_gradient(j, sampler, tau)
        return res

    def _prob_gradient(self, j, sampler, tau):
        l = random.randint(1, self.N)
        former = self._to_time_evolution(sampler.sample(l - 1), tau)
        latter = self._to_time_evolution(sampler.sample(self.N - l), tau)
        j_evolution = self._to_time_evolution(sampler.get(j), tau)[0]
        one_evolution = self._to_time_evolution(sampler.sample(1), tau)[0]

        def prepare_first():
            qc = init_circuit(self.nqubit, self.tool)
            for o in former:
                o.add_circuit(qc)
            j_evolution.add_circuit(qc)
            for o in latter:
                o.add_circuit(qc)
            return qc

        def prepare_second():
            qc = init_circuit(self.nqubit, self.tool)
            for o in former:
                o.add_circuit(qc)
            one_evolution.add_circuit(qc)
            for o in latter:
                o.add_circuit(qc)
            return qc

        return (self.mes_method.get_value(prepare_first, self.ntotal)
                - self.mes_method.get_value(prepare_second, self.ntotal)) / tau

    def _operator_gradient(self, sampler, tau):
        l = random.randint(1, self.N)
        ancilla = self.nqubit
        former = self._to_time_evolution(sampler.sample(l), tau)
        pauli = sampler.sample(1)
        latter = self._to_time_evolution(sampler.sample(self.N - l), tau)

        def prepare_first():
            qc = init_circuit(self.nqubit + 1, self.tool)
            for o in former:
                o.add_circuit(qc)
            self._add_swift_gate(qc, pauli, 0, ancilla)
            for o in latter:
                o.add_circuit(qc)

        def prepare_second():
            qc = init_circuit(self.nqubit + 1, self.tool)
            for o in former:
                o.add_circuit(qc)
            self._add_swift_gate(qc, pauli, 1, ancilla)
            for o in latter:
                o.add_circuit(qc)

        return self.ancilla_mes_method.get_value(prepare_first, self.ntotal) - self.ancilla_mes_method.get_value(
            prepare_second, self.ntotal)

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
