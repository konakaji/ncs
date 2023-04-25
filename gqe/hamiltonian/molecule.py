import logging
from qwrapper.hamiltonian import Hamiltonian
from openfermion.chem import MolecularData, geometry_from_pubchem
from openfermion.transforms import get_fermion_operator, bravyi_kitaev, jordan_wigner
from openfermionpyscf import run_pyscf

from gqe.util import parse


class MolecularHamiltonian(Hamiltonian):
    def __init__(self, nqubit, basis, pubchem_name, bravyi_kitaev=True):
        self.identity_coeff = 0
        self.bravyi_kitaev = bravyi_kitaev
        hamiltonian = self._get_molecule_hamiltonian(basis, pubchem_name)
        coeffs, paulis, identity_coeff = parse(hamiltonian, nqubit)
        self.identity_coeff = identity_coeff
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
