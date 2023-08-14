from gqe.operator_pool.op import ListablePool
import tequila as tq

class UCCSD(ListablePool):
    def __init__(self, atom1type, atom2type, bond_length, basis_set,
                 method='MP2', threshold=1.e-6, trotter_steps=1):
        active_orbitals = {'A1': [1], "B1": [0]}
        geometry = (f"{atom1type} 0.0 0.0 0.0\n" +
                    f"{atom2type} 0.0 0.0 {bond_length}")
        molecule = tq.chemistry.Molecule(geometry=geometry,
                                          basis_set=basis_set,
                                          active_orbitals=active_orbitals)
        H = molecule.make_hamiltonian()
        U = molecule.make_uccsd_ansatz(initial_amplitudes=method,
                                       threshold=threshold,
                                       trotter_steps=trotter_steps)
        self.molecular_hamiltonian = H
        self.uccsd_operator = U

    def all(self):
        return self.molecular_hamiltonian, self.uccsd_operator
