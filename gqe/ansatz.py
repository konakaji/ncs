from qwrapper.obs import PauliObservable
from qwrapper.operator import ControllablePauli


class Ansatz:
    def __init__(self, h_vec, o_vec: [PauliObservable]):
        self.h_vec = h_vec
        self._o_vec = o_vec

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

    def lam(self):
        res = 0
        for h in self.get_positive_h_vec():
            res += h
        return res

    def copy(self):
        return Ansatz(self.h_vec, self._o_vec)
