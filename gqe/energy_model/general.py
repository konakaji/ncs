from typing import Any

import torch, math
import torch.optim as optim
import pytorch_lightning as pl
from qwrapper.obs import PauliObservable
from qswift.qswift import OperatorPool
from gqe.energy_estimator.general import GeneralEstimator
from gqe.energy_model.sampler import NaiveSampler


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


class GeneralEnergyModel(pl.LightningModule):
    def __init__(self, sampler: NaiveSampler,
                 pool: OperatorPool,
                 estimator: GeneralEstimator, N, n_sample=10,
                 lr=1e-4,
                 beta1=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.estimator = estimator
        self.network = sampler.nn
        self.sampler = sampler
        self.N = N
        self.n_sample = n_sample
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
        for _ in batch:
            indices = self.sampler.sample_indices(count=self.N * self.n_sample)
            fs = self.sampler.nn.forward([self._pool.get(index) for index in indices]).reshape(self.n_sample, self.N)
            sum_fs = torch.sum(fs, 1)
            mean = sum_fs.mean()
            gs = []
            for chunk in indices.reshape(self.n_sample, self.N):
                gs.append(self.estimator.evaluate(chunk))
            value = -torch.dot(torch.flatten(sum_fs) - mean, torch.tensor(gs, dtype=torch.float32)) / (
                    len(gs) * math.sqrt(self.N))
            if loss is None:
                loss = value
            else:
                loss += value
        return loss