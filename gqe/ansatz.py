from qwrapper.obs import PauliObservable


class Ansatz:
    def __init__(self, h_vec, o_vec: [PauliObservable]):
        self._h_vec = h_vec
        self._o_vec = o_vec

    def get_positive_h_vec(self):
        return [abs(h) for h in self._h_vec]

    def get_signed_o_vec(self):
        rs = []
        for j, h in enumerate(self._h_vec):
            if h < 0:
                rs.append(self._o_vec[j])
        return rs

    def lam(self):
        res = 0
        for h in self.get_positive_h_vec():
            res += h
        return res

    def update(self, h_vec):
        self._h_vec = h_vec
