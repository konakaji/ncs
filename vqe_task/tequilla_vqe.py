from tequila import Molecule
from gqe.operator_pool.uccsd import generate_molecule

import tequila as tq


class VQE:
    def __init__(self, molecule: Molecule):
        self.molecule = molecule

    def run(self, maxiter=100):
        U = self.molecule.make_uccsd_ansatz(threshold=1e-6, trotter_steps=1)
        H = self.molecule.make_hamiltonian()
        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(objective=E, method="BFGS", initial_values=0.0, maxiter=maxiter)
        print(result.energy)


if __name__ == '__main__':
    molecule = generate_molecule("N", "N", 1.8, "sto-3g", active_orbitals=[4, 5, 6, 7, 8, 9],
                                 bravyi_kitaev=False)
    print(VQE(molecule).run(maxiter=1))
