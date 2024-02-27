from experiment.experiment import LiHExperiment
from task.lih import get_lih_configs
from torch.utils.data import DataLoader
from gqe.mingpt.data import EnergyDataset

cfg = get_lih_configs()

loader = DataLoader(EnergyDataset(['checkpoints/gptqe/run_1016_12_13/trajectory_1.0.ckpt']), batch_size=50)

LiHExperiment().pretrain(cfg, loader)
