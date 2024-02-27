from experiment.experiment import LiHExperiment
from experiment.circuit import OptimalCircuitWriter
from gqe.gptqe.data import EnergyDataset
from task.lih import get_lih_configs

cfg = get_lih_configs()

writer = OptimalCircuitWriter(LiHExperiment())
dataset = EnergyDataset(['checkpoints/gptqe/run_1016_12_13/trajectory_1.0.ckpt'])
print(writer.write_in_string(dataset, cfg))
