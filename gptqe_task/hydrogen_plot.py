from experiment.experiment import HydrogenExperiment
from experiment.configs import get_default_configs


def get_hydrogen_cfg():
    cfg = get_default_configs()
    cfg.distances = [0.5, 0.6, 0.7, 0.7414, 0.8, 0.9, 1.0,
                     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    cfg.ngates = 10
    cfg.max_iters = 20
    cfg.n_electrons = 2
    cfg.verbose = False
    cfg.temperature = 5
    cfg.molecule_name = "H2"
    return cfg


if __name__ == '__main__':
    m = {}
    with open("../output/H2_min.txt") as f:
        for l in f.readlines():
            dist, energy = l.rstrip().split("\t")
            m[float(dist)] = float(energy)
    cfg = get_hydrogen_cfg()
    HydrogenExperiment().plot_figure(cfg, [m[dist] for dist in cfg.distances])

    m = {}
    e = {}
    with open("../output/H2_avg.txt") as f:
        for l in f.readlines():
            dist, energy, error = l.rstrip().split("\t")
            m[float(dist)] = float(energy)
            e[float(dist)] = float(error)
    cfg = get_hydrogen_cfg()
    HydrogenExperiment().plot_figure(cfg,
                                     [m[dist] for dist in cfg.distances],
                                     [e[dist] for dist in cfg.distances])
