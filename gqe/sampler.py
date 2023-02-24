from qwrapper.sampler import DefaultImportantSampler
from gqe.ansatz import Ansatz
from gqe.util import identity


class OperatorSampler:
    def __init__(self, ansatz: Ansatz, lam, nqubit):
        hs = ansatz.get_positive_h_vec()
        hs.append(lam - sum(hs))
        self.hs = hs
        operators = ansatz.get_signed_o_vec()
        operators.append(identity(nqubit))
        self.operators = operators
        self.sampler = DefaultImportantSampler(self.hs)

    def sample(self, count=1):
        res = []
        for j in range(count):
            index = self.sampler.sample_index()
            res.append(self.operators[index])
        return res

    def get(self, j):
        return self.operators[j]
