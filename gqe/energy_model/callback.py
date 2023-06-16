import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Callback
from qwrapper.hamiltonian import Hamiltonian
from qswift.compiler import OperatorPool
from gqe.energy_model.sampler import NaiveSampler
from gqe.energy_estimator.iid import IIDEstimator
from gqe.energy_estimator.qdrift import QDriftEstimator
from gqe.energy_estimator.general import GeneralEstimator


class RecordEnergy(Callback):
    def __init__(self, sampler: NaiveSampler, estimator: QDriftEstimator, n_samples):
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


class IIDRecordEnergy(Callback):
    def __init__(self, sampler: NaiveSampler, estimator: IIDEstimator, pool: OperatorPool,
                 lam):
        self.sampler = sampler
        self.estimator = estimator
        self.pool = pool
        self.lam = lam
        self.records = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        energy = self.estimator.value(self.sampler, self.pool, self.lam)
        self.records.append(energy)
        print(energy)
        super().on_train_epoch_end(trainer, pl_module)

    def save(self, path):
        with open(path, 'w') as f:
            for j, record in enumerate(self.records):
                f.write(f'{j}\t{record}\n')


class V2ExactRecordEnergy(Callback):
    def __init__(self, sampler: NaiveSampler, estimator: GeneralEstimator, lam, N, n_samples=0):
        self.sampler = sampler
        self.estimator = estimator
        self.lam = lam
        self.N = N
        self.n_samples = n_samples
        self.records = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.n_samples == 0:
            probs = self.sampler.all_probabilities()
            hs = self.lam * probs / sum(probs)
            print([h for h in hs])
            paulis = [self.sampler.all_paulis[index] for index in range(len(probs))]
            value = self.estimator.exact(Hamiltonian(hs, paulis, paulis[0].nqubit))
        else:
            values = []
            for j in range(self.n_samples):
                indices = self.sampler.sample_indices(self.N)
                values.append(self.estimator.evaluate(indices))
            value = np.mean(values)
        print(value)
        self.records.append(value)
        super().on_train_epoch_end(trainer, pl_module)

    def save(self, path):
        with open(path, 'w') as f:
            for j, record in enumerate(self.records):
                f.write(f'{j}\t{record}\n')
