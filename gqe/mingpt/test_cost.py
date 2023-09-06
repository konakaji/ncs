from unittest import TestCase
from qwrapper.hamiltonian import HeisenbergModel
from qwrapper.operator import PauliObservable
from qswift.initializer import ZBasisInitializer
from qswift.compiler import DefaultOperatorPool
from gqe.mingpt.cost import EnergyCost
import torch


class TestCost(TestCase):
    def test_energy(self):
        hamiltonian = HeisenbergModel(3)
        initializer = ZBasisInitializer()
        pool = DefaultOperatorPool([PauliObservable("IYX"), PauliObservable("ZXI")])
        cost = EnergyCost(hamiltonian, initializer, pool, [0.1, 0.2])

        print(cost.energy([[-1, 0, 1, 1, 2]]))
