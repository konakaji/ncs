from typing import Any

import torch, math
import torch.optim as optim
import pytorch_lightning as pl
from qwrapper.obs import PauliObservable
from qswift.qswift import OperatorPool, DefaultOperatorPool
from qwrapper.sampler import ImportantSampler, FasterImportantSampler
from gqe.energy_estimator.qswift import SecondQSwiftEstimator
from pytorch_lightning import Callback
import numpy as np


def all(nqubit):
    results = [""]
    for _ in range(nqubit):
        next = []
        for pauli in ["I", "X", "Y", "Z"]:
            for r in results:
                r = r + pauli
                next.append(r)
        results = next
    paulis = []
    for r in results:
        paulis.append(PauliObservable(r))
    for r in results:
        paulis.append(PauliObservable(r, -1))
    return paulis


class RecordEnergy(Callback):
    def __init__(self, model: ):
        self.model = model
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


class NaiveSampler(ImportantSampler):
    def __init__(self, nn, operator_pool: DefaultOperatorPool, beta):
        self.nn = nn
        self.beta = beta
        self.all_paulis = operator_pool.paulis
        self.sampler = self.reset()

    def sample_index(self):
        return self.sample_indices(1)[0]

    def sample_indices(self, count=1):
        return self.sampler.sample_indices(count)

    def reset(self):
        probs = self._all_probabilities()
        self.sampler = FasterImportantSampler(probs)
        return self.sampler

    def _all_probabilities(self):
        results = []
        for f in self.nn.forward(self.all_paulis):
            results.append(math.exp(-self.beta * f))
        return results


class EnergyModel(pl.LightningModule):
    def __init__(self, sampler: NaiveSampler,
                 pool: OperatorPool,
                 estimator: SecondQSwiftEstimator, lam, nqubit, n_samples, lr=1e-4,
                 beta1=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.network = sampler.nn
        self.estimator = estimator
        self.n_samples = n_samples
        self.sampler = sampler
        self.lam = lam
        self.nqubit = nqubit
        self._pool = pool

    def forward(self, paulis) -> Any:
        value = self.network.forward(paulis)
        return value

    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)  # Exponential decay over epochs
        return [optimizer], [scheduler]

    def training_step(self, batch):
        self.sampler.reset()
        loss = None
        total_indices = []
        for _ in batch:
            indices = self.sampler.sample_indices(1)
            gs = []
            for index in indices:
                g = self.estimator.grad(self.sampler, self._pool, self.lam, index)
                gs.append(g)
            total_indices.extend(indices)
            fs = self.sampler.nn.forward([self._pool.get(index) for index in indices])
            mean = fs.mean()
            value = -torch.dot(torch.flatten(fs) - mean, torch.tensor(gs, dtype=torch.float32)) / len(indices)
            if loss is None:
                loss = value
            else:
                loss += value
        return loss
