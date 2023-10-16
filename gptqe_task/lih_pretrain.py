from experiment.experiment import LiHExperiment
from experiment.configs import get_default_pretrain_configs
from torch.utils.data import DataLoader
from gqe.mingpt.data import EnergyDataset

cfg = get_default_pretrain_configs()
cfg.ngates = 40
cfg.num_samples = 50
cfg.n_electrons = 2
cfg.energy_offset = 7
cfg.nqubit = 10
cfg.molecule_name = "LiH"
cfg.verbose = True

loader = DataLoader(EnergyDataset(['checkpoints/gptqe/run_1016_12_13/trajectory_1.0.ckpt']), batch_size=50)

LiHExperiment().pretrain(cfg, loader)
