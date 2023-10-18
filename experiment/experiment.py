from experiment.base import GPTQEBase
from gqe.operator_pool.uccsd import generate_molecule, do_generate_molecule


class HydrogenExperiment(GPTQEBase):

    def get_molecule(self, distance, cfg):
        return generate_molecule("H", "H", distance, "sto-3g", bravyi_kitaev=cfg.is_bravyi)


class LiHExperiment(GPTQEBase):
    def get_molecule(self, distance, cfg):
        return generate_molecule("Li", "H", distance, "sto-3g", bravyi_kitaev=cfg.is_bravyi)


class BeH2Experiment(GPTQEBase):
    def get_molecule(self, distance, cfg):
        geometry = f"H 0.0 0.0 0.0\n" + f"Be 0.0 0.0 {distance}\n" + f"H 0.0 0.0 {2 * distance}\n"
        return do_generate_molecule(geometry, "sto-3g", bravyi_kitaev=False)
