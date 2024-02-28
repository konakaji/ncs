from experiment.experiment import O2Experiment
from experiment.configs import get_default_configs


def get_o2_configs():
    cfg = get_default_configs()
    cfg.distances = [1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9]
    cfg.ngates = 60
    cfg.max_iters = 500
    cfg.num_samples = 50
    cfg.n_electrons = 8
    cfg.energy_offset = 147
    cfg.nqubit = 12
    cfg.del_temperature = 0.1
    cfg.temperature = 20
    cfg.molecule_name = "O2"
    return cfg


if __name__ == '__main__':
    cfg = get_o2_configs()
    for seed in [1, 2, 3]:
        cfg.seed = seed
        O2Experiment().train(cfg)