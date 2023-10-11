from experiment.experiment import LiHExperiment
from experiment.configs import get_default_configs

cfg = get_default_configs()
cfg.distances = [1.0, 1.5, 1.57, 2.0, 2.5, 3.0]
cfg.ngates = 40
cfg.max_iters = 500
cfg.num_samples = 50
cfg.n_electrons = 2
cfg.energy_offset = 7
cfg.nqubit = 10
cfg.molecule_name = "LiH"

LiHExperiment().run(cfg)
