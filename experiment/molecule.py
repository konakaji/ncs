import logging

from qwrapper.circuit import QWrapper
from qwrapper.hamiltonian import Hamiltonian
from qwrapper.obs import PauliObservable
from openfermion import FermionOperator
from openfermion.chem import MolecularData, geometry_from_pubchem
from openfermion.transforms import get_fermion_operator, bravyi_kitaev, jordan_wigner
from openfermionpyscf import run_pyscf
import tequila as tq

logger = logging.getLogger('benchmark.molecule')


def parse(operator: FermionOperator, nqubit):
    coeffs = []
    paulis = []
    identity_coeff = 0
    logger.info("Starting to parse FermionOperator using " + str(nqubit) + " qubits...")
    for t in operator:
        logger.info("Operator t: ", t)
        for term, coefficient in t.terms.items():
            logger.info("Term, coeff: ", term, coefficient)
            dict = {}
            for index, p_char in term:
                logger.info("Index, p_char: ", index, p_char)
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
                paulis.append(PauliObservable(p_string))
            else:
                identity_coeff += coefficient.real
    return coeffs, paulis, identity_coeff


def generate_molecule(atom1type, atom2type, bond_length, basis_set, active_orbitals=None):
    geometry = (f"{atom1type} 0.0 0.0 0.0\n" +
                f"{atom2type} 0.0 0.0 {bond_length}")
    if active_orbitals is not None:
        return tq.chemistry.Molecule(geometry=geometry,
                                     basis_set=basis_set,
                                     active_orbitals=active_orbitals)
    return tq.chemistry.Molecule(geometry=geometry, basis_set=basis_set)


class MolecularHamiltonian(Hamiltonian):
    def __init__(self, nqubit, basis, pubchem_name, bravyi_kitaev=True):
        self.bravyi_kitaev = bravyi_kitaev
        hamiltonian = self._get_molecule_hamiltonian(basis, pubchem_name)
        coeffs, paulis, identity_coeff = parse(hamiltonian, nqubit)
        self.identity = identity_coeff
        logging.log(logging.INFO, f"# of terms is {len(coeffs)}")
        super().__init__(coeffs, paulis, nqubit)

    def _get_molecule_hamiltonian(self, basis, pubchem_name):
        geometry = geometry_from_pubchem(pubchem_name)
        multiplicity = 1
        charge = 0
        molecule = MolecularData(geometry, basis, multiplicity, charge)
        logging.info("start running pyscf")
        molecule = run_pyscf(molecule, run_scf=0, run_fci=0)
        logging.info("finish running pyscf.")
        molecule_hamiltonian = molecule.get_molecular_hamiltonian()
        logging.info("start qubit mapping.")
        if self.bravyi_kitaev:
            result = bravyi_kitaev(get_fermion_operator(molecule_hamiltonian))
        else:
            result = jordan_wigner(get_fermion_operator(molecule_hamiltonian))
        logging.info("finish qubit mapping.")
        return result


class DiatomicMolecularHamiltonian(Hamiltonian):
    def __init__(self, nqubit, molecule: tq.quantumchemistry.Molecule = None, bravyi_kitaev=True, **kwargs):
        self.bravyi_kitaev = bravyi_kitaev

        if molecule is None:
            basis = kwargs["basis"]
            atom1_type = kwargs["atom1_type"]
            atom2_type = kwargs["atom2_type"]
            diatomic_bond_length = kwargs["diatomic_bond_length"]
            hamiltonian = self._get_molecule_hamiltonian(basis, atom1_type, atom2_type, diatomic_bond_length)
        else:
            if bravyi_kitaev:
                transformation = molecule._initialize_transformation("BravyiKitaev")
            else:
                transformation = molecule._initialize_transformation("JordanWigner")
            molecule.transformation = transformation
            hamiltonian = [molecule.make_hamiltonian().qubit_operator]
        coeffs, paulis, identity_coeff = parse(hamiltonian, nqubit)
        logging.log(logging.INFO, f"# of terms is {len(coeffs)}")
        logging.log(logging.INFO, f"Identity coefficient is {identity_coeff}")
        super().__init__(coeffs, paulis, nqubit)
        self._identity = identity_coeff

    def _get_molecule_hamiltonian(self, basis, atom1_type, atom2_type, diatomic_bond_length):
        geometry = [(atom1_type, (0., 0., 0.)), (atom2_type, (0., 0., diatomic_bond_length))]
        multiplicity = 1
        charge = 0
        molecule = MolecularData(geometry, basis, multiplicity, charge)
        logging.info("start running pyscf")
        molecule = run_pyscf(molecule, run_scf=0, run_fci=0)
        logging.info("finish running pyscf.")
        molecule_hamiltonian = molecule.get_molecular_hamiltonian()
        logging.info("start qubit mapping.")
        if self.bravyi_kitaev:
            result = bravyi_kitaev(get_fermion_operator(molecule_hamiltonian))
        else:
            result = jordan_wigner(get_fermion_operator(molecule_hamiltonian))
        logging.info("finish qubit mapping.")
        return result
