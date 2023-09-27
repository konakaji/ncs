from gqe.operator_pool.op import ListablePool
from qwrapper.operator import ControllablePauli
import tequila as tq


class UCCSD(ListablePool):
    def __init__(self, nqubit, molecule=None, method='MP2', threshold=1.e-6, trotter_steps=1,
                 **kwargs):
        if molecule is None:
            atom1type = kwargs["atom1type"]
            atom2type = kwargs["atom2type"]
            bond_length = kwargs["bond_length"]
            basis_set = kwargs["basis_set"]
            molecule = generate_molecule(atom1type, atom2type, bond_length, basis_set)
        u = molecule.make_uccsd_ansatz(initial_amplitudes=method,
                                       threshold=threshold,
                                       trotter_steps=trotter_steps)
        p_strings = set()
        for g in u.gates:
            p_array = ["I"] * nqubit
            for p in g.generator.paulistrings:
                for k, v in p.items():
                    p_array[k] = v
                p_string = "".join(p_array)
                p_strings.add(p_string)
        paulis = []
        for p_string in sorted(p_strings):
            paulis.append(ControllablePauli(p_string))
        self.nqubit = nqubit
        self.paulis = paulis

    def size(self):
        return len(self.paulis)

    def all(self):
        return self.paulis


def generate_molecule(atom1type, atom2type, bond_length, basis_set, active_orbitals=None, bravyi_kitaev=True):
    geometry = (f"{atom1type} 0.0 0.0 0.0\n" +
                f"{atom2type} 0.0 0.0 {bond_length}")
    transformation = "jordan-wigner"
    if bravyi_kitaev:
        transformation = "bravyi-kitaev"
    if active_orbitals is not None:
        return tq.chemistry.Molecule(geometry=geometry,
                                     basis_set=basis_set,
                                     active_orbitals=active_orbitals, transformation=transformation)
    return tq.chemistry.Molecule(geometry=geometry, basis_set=basis_set)


def do_generate_molecule(geometry, basis_set, active_orbitals=None, bravyi_kitaev=True):
    transformation = "jordan-wigner"
    if bravyi_kitaev:
        transformation = "bravyi-kitaev"
    if active_orbitals is not None:
        return tq.chemistry.Molecule(geometry=geometry,
                                     basis_set=basis_set,
                                     active_orbitals=active_orbitals, transformation=transformation)
    return tq.chemistry.Molecule(geometry=geometry, basis_set=basis_set)
