import abc
from abc import ABC, abstractmethod
from qwrapper.operator import ControllablePauli


class OperatorPool(abc.ABC):
    @abstractmethod
    def all(self):
        pass


class AllPauliOperator(OperatorPool):
    def __init__(self, nqubit):
        self.nqubit = nqubit

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
