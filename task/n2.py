from experiment.experiment import N2Experiment
from experiment.configs import get_default_configs
import sys

def get_n2_configs():
    cfg = get_default_configs()
    cfg.distances = [0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8]
    cfg.ngates = 40
    cfg.max_iters = 500
    cfg.num_samples = 50
    cfg.n_electrons = 6 
    cfg.energy_offset = 106
    cfg.nqubit = 12
    cfg.del_temperature = 0.1
    cfg.molecule_name = "N2"
    #cfg.dry = True
    return cfg


if __name__ == '__main__':
    seed = int(sys.argv[1])
    cfg = get_n2_configs()
    cfg.seed = seed
    N2Experiment().train(cfg)