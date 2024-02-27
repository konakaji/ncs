from qswift.initializer import CircuitInitializer
from qwrapper.circuit import init_circuit
import numpy as np

class HFStateInitializer(CircuitInitializer):
    def __init__(self, n_electrons):
        self.n_electrons = n_electrons

    def init_circuit(self, nqubit, ancilla, tool):
        qc = init_circuit(nqubit, tool)
        count = 0
        for j in range(nqubit):
            if j in ancilla:
                continue
            qc.x(j)
            count += 1
            if count == self.n_electrons:
                break
        return qc

    def initial_state(self, dim):
        result = np.diag([0] * dim)
        index_pos = 0
        for j in range(self.n_electrons):
            index_pos += pow(2, j)
        result[index_pos][index_pos] = 1
        return result
