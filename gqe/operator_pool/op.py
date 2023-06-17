from abc import abstractmethod
from qwrapper.operator import ControllablePauli
from qswift.compiler import OperatorPool


class ListablePool(OperatorPool):
    @abstractmethod
    def all(self):
        pass


class AllPauliOperators(ListablePool):
    def __init__(self, nqubit):
        self.nqubit = nqubit
        self.paulis = self.all()

    def get(self, j):
        return self.paulis[j]

    def all(self):
        results = [""]
        for _ in range(self.nqubit):
            next = []
            for pauli in ["I", "X", "Y", "Z"]:
                for r in results:
                    r = r + pauli
                    next.append(r)
            results = next
        paulis = []
        for r in results:
            paulis.append(ControllablePauli(r))
        for r in results:
            paulis.append(ControllablePauli(r, -1))
        return paulis
