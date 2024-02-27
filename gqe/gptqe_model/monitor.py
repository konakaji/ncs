import sys
from gqe.gptqe_model.data import TrajectoryData


class FileMonitor:
    def __init__(self):
        self.lines = []

    def record(self, iter_num, loss, energies, indices):
        energies = energies.cpu().numpy().tolist()
        indices = indices.cpu().numpy().tolist()
        data = TrajectoryData(iter_num, loss.item(), indices, energies)
        self.lines.append(data.to_json())

    def save(self, path):
        with open(path, 'w') as f:
            for l in self.lines:
                f.write(f"{l}\n")
