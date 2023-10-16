from experiment.experiment import LiHExperiment
from experiment.configs import get_default_configs
from experiment.circuit import OptimalCircuitWriter
from gqe.gptqe.data import EnergyDataset

cfg = get_default_configs()
cfg.distances = [1.0, 1.5, 1.57, 2.0, 2.5, 3.0]
cfg.ngates = 40
cfg.max_iters = 500
cfg.num_samples = 50
cfg.n_electrons = 2
cfg.energy_offset = 7
cfg.nqubit = 10
cfg.molecule_name = "LiH"

writer = OptimalCircuitWriter(LiHExperiment())
dataset = EnergyDataset(['checkpoints/gptqe/run_1016_12_13/trajectory_1.0.ckpt'])
print(writer.write_in_string(dataset, cfg))
