import sys
from experiment.configs import get_default_configs
from experiment.experiment import N2Experiment
from torch.utils.data import DataLoader
from gqe.mingpt.data import EnergyDataset


def get_n2_configs():
    cfg = get_default_configs()
    cfg.distance = 1.4
    cfg.ngates = 40
    cfg.num_samples = 50
    cfg.n_electrons = 6
    cfg.energy_offset = 106
    cfg.nqubit = 12
    cfg.temperature = 5
    cfg.del_temperature = 0.1
    cfg.molecule_name = "N23"
    cfg.max_iters = 1
    cfg.save_dir = "../output/"
    cfg.vocab_size = 1390
    cfg.verbose = True
    cfg.resid_pdrop = 0.2
    cfg.embd_pdrop = 0.2
    cfg.attn_pdrop = 0.2
    # cfg.dry = True
    return cfg


if __name__ == '__main__':
    seed = sys.argv[1]
    cfg = get_n2_configs()
    cfg.seed = int(seed)
    cfg.max_iters = 1
    cfg.temperature = 5
    loader = DataLoader(EnergyDataset(['../output/N2_transfer_1.2_to_1.4.ckpt'], threshold=-107.45), batch_size=50)
    path = N2Experiment().pretrain(cfg, loader)
    cfg.check_points = {"1_4": path}
    cfg.resid_pdrop = 0
    cfg.embd_pdrop = 0
    cfg.attn_pdrop = 0
    cfg.max_iters = 500
    cfg.temperature = 5 
    N2Experiment().train_single(cfg)
