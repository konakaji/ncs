from experiment.experiment import O2Experiment
from experiment.configs import get_default_configs


def get_o2_configs():
    cfg = get_default_configs()
    cfg.distances = [1.0, 1.5, 1.57, 2.0, 2.5, 3.0]
    cfg.ngates = 40
    cfg.max_iters = 500
    cfg.num_samples = 50
    cfg.n_electrons = 6
    cfg.energy_offset = 147
    cfg.nqubit = 12
    cfg.del_temperature = 0.1
    cfg.molecule_name = "O2"
    return cfg


if __name__ == '__main__':
    O2Experiment().run(get_o2_configs())
