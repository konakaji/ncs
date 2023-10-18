from experiment.experiment import HydrogenExperiment
from experiment.configs import get_default_configs


def get_hydrogen_cfg():
    cfg = get_default_configs()
    cfg.distances = [0.5, 0.6, 0.7, 0.7414, 0.8, 0.9, 1.0, 1.5, 2.0]
    cfg.ngates = 20
    cfg.n_electrons = 2
    cfg.verbose = True
    cfg.temperature = 5
    cfg.molecule_name = "H2"
    return cfg


if __name__ == '__main__':
    HydrogenExperiment().run(get_hydrogen_cfg())
