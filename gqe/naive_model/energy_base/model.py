from typing import Any

import torch
import torch.optim as optim
import pytorch_lightning as pl

from gqe.energy_model.sampler import NaiveSampler
from gqe.energy_estimator.qdrift import QDriftEstimator

"""
This package is deprecated. 
"""


class EnergyModel(pl.LightningModule):
    def __init__(self, sampler: NaiveSampler, estimator: QDriftEstimator, n_samples, lr=1e-4, beta1=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.network = sampler.nn
        self.estimator = estimator
        self.n_samples = n_samples
        self.sampler = sampler

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
            gs, indices = self.estimator.grads(self.sampler)
            total_indices.extend(indices)
            fs = self.sampler.nn.forward(self.sampler.get_all(indices))
            mean = fs.mean()
            value = -torch.dot(torch.flatten(fs) - mean, torch.tensor(gs, dtype=torch.float32)) / len(indices)
            if loss is None:
                loss = value
            else:
                loss += value
        return loss
