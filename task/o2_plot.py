from experiment.experiment import O2Experiment
from experiment.configs import get_default_configs


def get_o2_configs():
    cfg = get_default_configs()
    cfg.distances = [1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9]
    cfg.ngates = 40
    cfg.max_iters = 500
    cfg.num_samples = 50
    cfg.n_electrons = 8
    cfg.energy_offset = 147
    cfg.nqubit = 12
    cfg.del_temperature = 0.1
    cfg.molecule_name = "O2"
    return cfg


if __name__ == '__main__':
    m = {}
    with open("../output/O2_min.txt") as f:
        for l in f.readlines():
            dist, energy = l.rstrip().split("\t")
            m[float(dist)] = float(energy)
    cfg = get_o2_configs()
    O2Experiment().plot_figure(cfg, [m[dist] for dist in cfg.distances])

    m = {}
    e = {}
    with open("../output/O2_avg.txt") as f:
        for l in f.readlines():
            dist, energy, error = l.rstrip().split("\t")
            m[float(dist)] = float(energy)
            e[float(dist)] = float(error)
    cfg = get_o2_configs()
    O2Experiment().plot_figure(cfg,
                                     [m[dist] for dist in cfg.distances],
                                     [e[dist] for dist in cfg.distances])

