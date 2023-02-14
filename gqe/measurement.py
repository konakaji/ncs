from qwrapper.measurement import MeasurementMethod
from qwrapper.obs import Hamiltonian, PauliObservable


class AncillaMeasurementMethod(MeasurementMethod):
    def __init__(self, hamiltonian: Hamiltonian):
        paulis = []
        for p in hamiltonian.paulis:
            paulis.append(PauliObservable(p.p_string + "X", p.sign))
        super().__init__(Hamiltonian(hamiltonian.hs, paulis, hamiltonian.nqubit))
