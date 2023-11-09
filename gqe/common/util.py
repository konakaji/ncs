from qwrapper.hamiltonian import Hamiltonian
import hashlib


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
