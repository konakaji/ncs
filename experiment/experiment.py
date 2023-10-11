from experiment.base import GPTQEBase
from gqe.operator_pool.uccsd import generate_molecule
from benchmark.molecule import DiatomicMolecularHamiltonian


class HydrogenExperiment(GPTQEBase):

    def get_molecule(self, distance, cfg):
        return generate_molecule("H", "H", distance, "sto-3g", bravyi_kitaev=cfg.is_bravyi)

    def get_hamiltonian(self, molecule, cfg):
        hamiltonian = DiatomicMolecularHamiltonian(cfg.nqubit, molecule, bravyi_kitaev=cfg.is_bravyi)
        return hamiltonian


class LiHExperiment(GPTQEBase):
    def get_hamiltonian(self, molecule, cfg):
        return DiatomicMolecularHamiltonian(cfg.nqubit, molecule, bravyi_kitaev=cfg.is_bravyi)

    def get_molecule(self, distance, cfg):
        return generate_molecule("Li", "H", distance, "sto-3g", bravyi_kitaev=cfg.is_bravyi)
