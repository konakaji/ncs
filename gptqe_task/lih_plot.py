from experiment.experiment import LiHExperiment
from experiment.configs import get_default_configs


def get_lih_configs():
    cfg = get_default_configs()
    cfg.distances = [1.0, 1.25, 1.5, 1.57, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    cfg.ngates = 40
    cfg.max_iters = 300
    cfg.num_samples = 50
    cfg.n_electrons = 2
    cfg.energy_offset = 7
    cfg.nqubit = 10
    cfg.del_temperature = 0.1
    cfg.molecule_name = "LiH"
    cfg.small = True
    return cfg


if __name__ == '__main__':
    m = {}
    with open("../output/LiH_min.txt") as f:
        for l in f.readlines():
            dist, energy = l.rstrip().split("\t")
            m[float(dist)] = float(energy)
    cfg = get_lih_configs()
    LiHExperiment().plot_figure(cfg, [m[dist] for dist in cfg.distances])

    m = {}
    e = {}
    with open("../output/LiH_avg.txt") as f:
        for l in f.readlines():
            dist, energy, error = l.rstrip().split("\t")
            m[float(dist)] = float(energy)
            e[float(dist)] = float(error)
    cfg = get_lih_configs()
    LiHExperiment().plot_figure(cfg,
                                     [m[dist] for dist in cfg.distances],
                                     [e[dist] for dist in cfg.distances])
