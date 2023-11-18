from experiment.experiment import N2Experiment
from experiment.configs import get_default_configs
from datetime import datetime

def get_n2_configs():
    cfg = get_default_configs()
    cfg.distances = [0.9, 1.0, 1.1, 1.2, 1.5, 1.8]
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
    N2Experiment().plot_figure(get_n2_configs(), [-107.21665954589844,-107.4986343383789,-107.59954071044922, -107.61727142333984, -107.522, -107.416])
