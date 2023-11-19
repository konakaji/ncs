from experiment.experiment import N2Experiment
from experiment.configs import get_default_configs
from datetime import datetime


def get_n2_configs():
    cfg = get_default_configs()
    cfg.distances = [0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.7]
    cfg.ngates = 40
    cfg.max_iters = 500
    cfg.num_samples = 50
    cfg.n_electrons = 6
    cfg.energy_offset = 106.5
    cfg.nqubit = 12
    cfg.del_temperature = 0.1
    cfg.molecule_name = "N2"
    cfg.dry = True
    cfg.save_dir = ""
    return cfg


if __name__ == '__main__':
    m = {}
    with open("../output/N2_min.txt") as f:
        for l in f.readlines():
            dist, energy = l.rstrip().split("\t")
            m[float(dist)] = float(energy)
    cfg = get_n2_configs()
    N2Experiment().plot_figure(cfg, [m[dist] for dist in cfg.distances])
