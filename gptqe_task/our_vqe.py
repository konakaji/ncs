import random

from tequila import Molecule
from gqe.operator_pool.uccsd import UCCSD
from gqe.vqa.initializer import InitializerDelegate
from gqe.common.initializer import HFStateInitializer
from qml.core.pqc import TimeEvolutionPQC
from qml.core.vqe import VQE, Energy
from qwrapper.optimizer import AdamOptimizer
from benchmark.molecule import generate_molecule
from benchmark.molecule import DiatomicMolecularHamiltonian


class CustomVQE:
    def __init__(self, molecule: Molecule, nqubit, n_electrons):
        self.molecule = molecule
        self.nqubit = nqubit
        self.n_electrons = n_electrons
        self.hamiltonian = DiatomicMolecularHamiltonian(nqubit, molecule, bravyi_kitaev=False)

    def run(self, seed, count=40):
        pqc = TimeEvolutionPQC(self.nqubit)
        energy = Energy(self.hamiltonian, self.nqubit, pqc)
        uccsd = UCCSD(nqubit=self.nqubit, molecule=self.molecule)
        paulis = uccsd.paulis
        random.seed(seed)
        random.shuffle(paulis)
        c = 0
        print(len(paulis))
        for p in paulis:
            if c == count:
                break
            pqc.add_time_evolution(p, 0)
            c += 1
        vqe = VQE(energy, InitializerDelegate(HFStateInitializer(self.n_electrons),
                                              self.nqubit), AdamOptimizer(maxiter=1000))
        vqe.exec()


if __name__ == '__main__':
    molecule = generate_molecule("O", "O", 1.3, "sto-3g", active_orbitals=[4, 5, 6, 7, 8, 9])
    CustomVQE(molecule, 12, 6).run(count=40)
