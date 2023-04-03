from unittest import TestCase
from gqe.energymodel import OneHotEncoder
from qwrapper.obs import PauliObservable


class TestOneHotEncoder(TestCase):
    def test_encode(self):
        encoder = OneHotEncoder()
        pauli = PauliObservable('XYZ', -1)
        print(encoder.encode(pauli))
