from experiment.experiment import *
from experiment.configs import get_default_configs
import sys

def get_beh2_configs():
    cfg = get_default_configs()
    cfg.distances = [1.0, 1.15, 1.3, 1.45, 1.6, 1.75, 1.9, 2.05, 2.2, 2.35, 2.5] 
    cfg.ngates = 40
    cfg.max_iters = 500
    cfg.num_samples = 50 
    cfg.backward_frequency = 1
    cfg.n_electrons = 4
    cfg.energy_offset = 14
    cfg.nqubit = 12
    cfg.del_temperature = 0.1
    cfg.molecule_name = "BeH2"
    return cfg


if __name__ == '__main__':
    seed = int(sys.argv[1])
    cfg = get_beh2_configs()
    cfg.seed = seed
    BeH2Experiment().train(cfg)
