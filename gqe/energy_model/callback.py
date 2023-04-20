import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Callback

from gqe.energy_model.sampler import NaiveSampler
from gqe.energy_estimator.ee import EnergyEstimator


class RecordEnergy(Callback):
    def __init__(self, sampler: NaiveSampler, estimator: EnergyEstimator, n_samples):
        self.sampler = sampler
        self.estimator = estimator
        self.n_samples = n_samples
        self.records = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        values = []
        for j in range(self.n_samples):
            values.append(self.estimator.value(self.sampler))
        mean = np.mean(values)
        self.records.append(mean)
        print(mean)
        super().on_train_epoch_end(trainer, pl_module)

    def save(self, path):
        with open(path, 'w') as f:
            for j, record in enumerate(self.records):
                f.write(f'{j}\t{record}\n')