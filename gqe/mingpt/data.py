import random

import torch
from torch.utils.data import Dataset
from gqe.util import get_device
import json
import sys


class TrajectoryData:
    def __init__(self, iter_num, loss, indices, energies):
        self.iter_num = iter_num
        self.loss = loss
        self.indices = indices
        self.energies = energies

    def to_json(self):
        map = {
            "iter": self.iter_num,
            "loss": self.loss,
            "indices": self.indices,
            "energies": self.energies
        }
        return json.dumps(map)

    @classmethod
    def from_json(self, string):
        if string.startswith('"'):
            string = string[1:len(string) - 1]
            string = string.replace("\\", "")
        map = json.loads(string)
        return TrajectoryData(map["iter"], map["loss"], map["indices"], map["energies"])


class EnergyDataset(Dataset):
    def __init__(self, file_paths, threshold=sys.maxsize):
        tensor_x = []
        tensor_y = []
        self.min_energy = sys.maxsize
        self.min_indices = None
        for file_path in file_paths:
            with open(file_path) as f:
                datum = []
                for l in f.readlines():
                    data = TrajectoryData.from_json(l.rstrip())
                    datum.append(data)
                for data in datum:
                    for indices, energy in zip(data.indices, data.energies):
                        if threshold < energy:
                            continue
                        if self.min_energy > energy:
                            self.min_energy = energy
                            self.min_indices = indices
                        result = [0]
                        result.extend(indices)
                        tensor_x.append(result)
                        tensor_y.append(energy)
        self.tensors = (
            torch.tensor(tensor_x, dtype=torch.int64),
            torch.tensor(tensor_y, dtype=torch.float)
        )

    def __getitem__(self, index):
        result = self.tensors[0][index], self.tensors[1][index]
        return result

    def __len__(self):
        return self.tensors[0].size(0)
