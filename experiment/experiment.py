from experiment.base import GPTQETaskBase
from gqe.operator_pool.uccsd import generate_molecule, do_generate_molecule


class HydrogenExperiment(GPTQETaskBase):

    def get_molecule(self, distance, is_bravyi):
        return generate_molecule("H", "H", distance, "sto-3g", bravyi_kitaev=is_bravyi)


class LiHExperiment(GPTQETaskBase):
    def get_molecule(self, distance, is_bravyi):
        return generate_molecule("Li", "H", distance, "sto-3g", bravyi_kitaev=is_bravyi)


class BeH2Experiment(GPTQETaskBase):
    def get_molecule(self, distance, is_bravyi):
        geometry = f"H 0.0 0.0 0.0\n" + f"Be 0.0 0.0 {distance}\n" + f"H 0.0 0.0 {2 * distance}\n"
        return do_generate_molecule(geometry, "sto-3g", bravyi_kitaev=is_bravyi)


class N2Experiment(GPTQETaskBase):
    def get_molecule(self, distance, is_bravyi):
        return generate_molecule("N", "N", distance, "sto-3g", active_orbitals=[4, 5, 6, 7, 8, 9],
                                 bravyi_kitaev=is_bravyi)


class O2Experiment(GPTQETaskBase):
    def get_molecule(self, distance, is_bravyi):
        return generate_molecule("O", "O", distance, "sto-3g", active_orbitals=[4, 5, 6, 7, 8, 9],
                                 bravyi_kitaev=is_bravyi)


class CO2Experiment(GPTQETaskBase):
    def get_molecule(self, distance, is_bravyi):
        geometry = f"O 0.0 0.0 0.0\n" + f"C 0.0 0.0 {distance}\n" + f"O 0.0 0.0 {2 * distance}\n"
        return do_generate_molecule(geometry, "sto-3g", active_orbitals=[6, 7, 8, 9, 10, 11, 12, 13, 14], bravyi_kitaev=is_bravyi)