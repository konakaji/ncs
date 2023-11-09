from experiment.experiment import *
from experiment.configs import get_default_configs


def get_beh2_configs():
    cfg = get_default_configs()
    cfg.distances = [2.5]
    cfg.ngates = 60
    cfg.max_iters = 500
    cfg.num_samples = 15
    cfg.backward_frequency = 6
    cfg.n_electrons = 4
    cfg.energy_offset = 14
    cfg.nqubit = 12
    cfg.del_temperature = 0.1
    cfg.molecule_name = "BeH2"
    return cfg


if __name__ == '__main__':
    BeH2Experiment().train(get_beh2_configs())
