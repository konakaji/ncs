from unittest import TestCase
from gqe.common.initializer import *
from qwrapper.hamiltonian import Hamiltonian, to_matrix_hamiltonian
from qwrapper.obs import PauliObservable
from qswift.initializer import XBasisInitializer
import numpy as np


class CircuitInitializerTest(TestCase):
    def test_initial_state(self):
        obs = Hamiltonian([0.2, 0.5], [PauliObservable("ZZZZ"), PauliObservable("IZZI")], 4)
        matrix = to_matrix_hamiltonian(obs)

        initializer = HFStateInitializer(2)
        qc = initializer.init_circuit(4, {}, "qiskit")

        exact = obs.exact_value(qc)

        rho = initializer.initial_state(16)
        result = np.trace(rho.dot(matrix)).real

        self.assertAlmostEquals(exact, result)
