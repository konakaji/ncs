from experiment.experiment import HydrogenExperiment
from experiment.configs import get_default_configs


def get_hydrogen_cfg():
    cfg = get_default_configs()
    cfg.distances = [0.5, 0.6, 0.7, 0.7414, 0.8, 0.9, 1.0,
                     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    cfg.ngates = 10
    cfg.max_iters = 50
    cfg.n_electrons = 2
    cfg.verbose = False
    cfg.temperature = 5
    cfg.nshot = 0
    cfg.tool = "qulacs"
    cfg.molecule_name = "H2"
    cfg.cache = False
    cfg.tool = "qiskit"
    return cfg


if __name__ == '__main__':
    cfg = get_hydrogen_cfg()
    for seed in [1, 2, 3]:
        cfg.seed = seed
        HydrogenExperiment().train(cfg)
