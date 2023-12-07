from gqe.gptqe.data import EnergyDataset
from experiment.configs import get_default_configs
from experiment.experiment import N2Experiment
from gqe.gptqe.data import TrajectoryData


def get_n2_configs():
    cfg = get_default_configs()
    cfg.distance = 1.4
    cfg.ngates = 40
    cfg.max_iters = 500
    cfg.num_samples = 50
    cfg.n_electrons = 6
    cfg.energy_offset = 106
    cfg.nqubit = 12
    cfg.del_temperature = 0.1
    cfg.molecule_name = "N2"
    cfg.save_data = True
    # cfg.dry = True
    return cfg


if __name__ == '__main__':
    dataset = EnergyDataset(["../output/N2_trajectory_1.6.ckpt"])
    cfg = get_n2_configs()
    exp = N2Experiment()
    cost = exp._construct_cost(cfg.distance, cfg, print_exact=False)
    datum = []
    tensors = []
    energies = []
    data = TrajectoryData(0, 0, tensors, energies)
    for tensor in dataset.tensors[0]:
        if len(tensors) == cfg.num_samples:
            tensors = []
            energies = []
            data = TrajectoryData(0, 0, tensors, energies)
            datum.append(data)
        tensors.append(tensor.cpu().numpy().tolist())
        energies.append(cost.energy([tensor[1:]]).item())
    with open("../output/N2_transfer_1.6_to_1.4.ckpt", "w") as f:
        for data in datum:
            f.write(f"{data.to_json()}\n")
