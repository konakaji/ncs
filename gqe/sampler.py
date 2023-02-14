from qwrapper.sampler import DefaultImportantSampler
from gqe.ansatz import Ansatz


class OperatorSampler:
    def __init__(self, ansatz: Ansatz):
        self.hs = ansatz.get_positive_h_vec()
        self.operators = ansatz.get_signed_o_vec()
        self.sampler = DefaultImportantSampler(self.hs)

    def sample(self, count=1):
        res = []
        for j in range(count):
            index = self.sampler.sample_index()
            res.append(self.operators[index])
        return res

    def get(self, j):
        return self.operators[j]
