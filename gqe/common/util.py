from qwrapper.operator import ControllablePauli
from qwrapper.hamiltonian import Hamiltonian
from openfermion import FermionOperator
from torch.utils.data import Dataset
import hashlib
import random, torch


def identity(nqubit):
    return ControllablePauli("".join(["I" for _ in range(nqubit)]))


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def to_hash(hamiltonian: Hamiltonian):
    # TODO very naive implementation
    value = 0
    integers = [13, 17, 31, 37]
    current = 0

    for h in hamiltonian.hs:
        value += h * integers[current]
        current += 1
        current = current % len(integers)

    string = ",".join([p.p_string for p in hamiltonian.paulis])
    string += str(value)
    return hashlib.sha256(string.encode('ascii')).hexdigest()

def parse(operator: FermionOperator, nqubit):
    coeffs = []
    paulis = []
    identity_coeff = 0
    for t in operator:
        for term, coefficient in t.terms.items():
            dict = {}
            for index, p_char in term:
                dict[index] = p_char
                if index > nqubit - 1:
                    raise AttributeError("nqubit is not correct.")
            results = []
            is_identity = False
            if len(dict) == 0:
                is_identity = True
            for q_index in range(nqubit):
                if q_index in dict:
                    results.append(dict[q_index])
                    continue
                results.append("I")
            if not is_identity:
                coeffs.append(coefficient.real)
                p_string = "".join(results)
                paulis.append(ControllablePauli(p_string))
            else:
                identity_coeff += coefficient.real
    return coeffs, paulis, identity_coeff


class VoidDataset(Dataset):
    def __init__(self, count=10):
        self.count = count

    def __getitem__(self, index):
        return [random.uniform(0, 1) for _ in range(self.count)]

    def __len__(self):
        return self.count
