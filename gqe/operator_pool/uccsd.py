import openfermion
from gqe.operator_pool.op import OperatorPool
from gqe.util import parse
from openfermion.transforms import bravyi_kitaev, jordan_wigner


class UCCSD(OperatorPool):
    def __init__(self, nqubit, bk=True):
        single_amplitudes = []
        for i in range(nqubit):
            for j in range(nqubit):
                single_amplitudes.append([[i, j], 1])

        double_amplitudes = []
        for i in range(nqubit):
            for j in range(nqubit):
                for k in range(nqubit):
                    for l in range(nqubit):
                        double_amplitudes.append([[i, j, k, l], 1])
        operator = openfermion.circuits.uccsd_generator(single_amplitudes, double_amplitudes, anti_hermitian=False)
        if bk:
            fo = bravyi_kitaev(operator, nqubit)
        else:
            fo = jordan_wigner(operator)
        _, operators, _ = parse(fo, nqubit)
        self.operators = operators

    def all(self):
        return self.operators
