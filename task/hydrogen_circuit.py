from experiment.experiment import HydrogenExperiment
from experiment.configs import get_default_configs


def get_hydrogen_cfg():
    cfg = get_default_configs()
    cfg.distances = [2.0]
    cfg.ngates = 10
    cfg.max_iters = 20
    cfg.n_electrons = 2
    cfg.verbose = False
    cfg.temperature = 5
    cfg.molecule_name = "H2"
    return cfg


if __name__ == '__main__':
    cfg = get_hydrogen_cfg()
    cfg.seed = 0
    exp = HydrogenExperiment()
    seq = exp.construct_cost(2.0, cfg).sequence
    seq.tool = "qiskit"
    seq.get_circuit([58, 53, 55, 63, 43, 58, 49, 56, 58, 59]).draw()
    import matplotlib.pyplot as plt

    plt.savefig("../output/H2_2_circuit.png")
