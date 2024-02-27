from experiment.experiment import N2Experiment
from experiment.configs import get_default_configs


def get_n2_configs():
    cfg = get_default_configs()
    cfg.distances = [0.9, 1.0, 1.1, 1.2, 1.5, 1.8]
    cfg.ngates = 80
    cfg.max_iters = 500
    cfg.num_samples = 50
    cfg.n_electrons = 4
    cfg.energy_offset = 106
    cfg.nqubit = 12
    cfg.del_temperature = 0.1
    cfg.molecule_name = "N2"
    return cfg


if __name__ == '__main__':
    N2Experiment().random_benchmark(get_n2_configs())
