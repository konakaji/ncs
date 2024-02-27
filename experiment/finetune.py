from vqe.initializer import VQAInitializer
from qswift.initializer import CircuitInitializer
from qswift.sequence import Sequence
from vqe.pqc import TimeEvolutionPQC
from qwrapper.circuit import init_circuit


class InitializerDelegate(VQAInitializer):
    def __init__(self, initializer: CircuitInitializer, nqubit, tool="qulacs"):
        self.initializer = initializer
        self.nqubit = nqubit
        self.tool = tool

    def initialize(self):
        return self.initializer.init_circuit(self.nqubit, [], self.tool)


def to_time_evolutions(sequence: Sequence, indices):
    evolutions = []
    for index in indices:
        evolutions.append(sequence.pool.get(index))
    return evolutions


def to_pqc(sequence: Sequence, indices):
    result = TimeEvolutionPQC(sequence.observable.nqubit)
    operators, taus = to_time_evolutions(sequence, indices)
    for operator, tau in zip(operators, taus):
        result.add_time_evolution(operator, tau)
    return result

