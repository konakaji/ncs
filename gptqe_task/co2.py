from experiment.experiment import CO2Experiment
from experiment.configs import get_default_configs


def get_co2_configs():
    cfg = get_default_configs()
    cfg.distances = [1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9]
    cfg.ngates = 60
    cfg.max_iters = 500
    cfg.num_samples = 50
    cfg.n_electrons = 10
    cfg.energy_offset = 184
    cfg.nqubit = 18
    cfg.temperature = 10
    cfg.del_temperature = 0.1
    cfg.molecule_name = "CO2"
    cfg.tool = "cudaq"
    cfg.print_exact = False
    return cfg


if __name__ == '__main__':
    cfg = get_co2_configs()
    cfg.seed = 1
    CO2Experiment().train(cfg)
