from qwrapper.operator import ControllablePauli


def identity(nqubit):
    return ControllablePauli("".join(["I" for _ in range(nqubit)]))
