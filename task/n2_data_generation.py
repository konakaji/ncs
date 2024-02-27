from experiment.experiment import N2Experiment
from experiment.configs import get_default_configs
import sys


def get_n2_configs():
    cfg = get_default_configs()
    cfg.distance = 1.6
    cfg.ngates = 40
    cfg.max_iters = 500
    cfg.num_samples = 50
    cfg.n_electrons = 6
    cfg.energy_offset = 106
    cfg.nqubit = 12
    cfg.del_temperature = 0.1
    cfg.molecule_name = "N2"
    cfg.save_data = True
    # cfg.dry = True
    return cfg


if __name__ == '__main__':
    seed = 0
    cfg = get_n2_configs()
    cfg.seed = seed
    N2Experiment().train_single(cfg)
