from experiment.experiment import N2Experiment
from experiment.configs import get_default_configs


def get_n2_configs():
    cfg = get_default_configs()
    cfg.distances = [1.0, 1.5, 1.57, 2.0, 2.5, 3.0]
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
    N2Experiment().run(get_n2_configs())
