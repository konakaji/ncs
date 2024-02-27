from experiment.base import GPTQETaskBase
from gqe.gptqe_model.data import EnergyDataset
from qwrapper.operator import PauliTimeEvolution
import matplotlib.pyplot as plt


class OptimalCircuitWriter:
    def __init__(self, base: GPTQETaskBase):
        self.base = base

    def write(self, data: EnergyDataset, cfg):
        cost = self.base.construct_cost(1, cfg)
        cost.sequence.tool = "qiskit"
        cost.sequence.get_circuit(data.min_indices).qc.draw(output="mpl", plot_barriers=True)

    def write_in_string(self, data: EnergyDataset, cfg):
        cost = self.base.construct_cost(1, cfg)
        result = []
        for index in data.min_indices:
            operator = cost.sequence.pool.get(index)
            assert isinstance(operator, PauliTimeEvolution)
            result.append(f"{operator.pauli}({operator.t})")
        return " ".join(result)

    def write_and_show(self, data: EnergyDataset, cfg):
        self.write(data, cfg)
        plt.show()

    def write_and_save(self, data: EnergyDataset, cfg, path):
        self.write(data, cfg)
        plt.savefig(path)
