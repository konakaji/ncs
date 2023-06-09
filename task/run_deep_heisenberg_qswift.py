import random

import torch, pytorch_lightning as pl
from torch.utils.data import DataLoader
from gqe.energy_model.model2 import EnergyModel, all
from gqe.energy_model.callback import RecordEnergy
from gqe.energy_model.model2 import NaiveSampler
from gqe.energy_model.network import PauliEnergy
from gqe.energy_estimator.qswift import SecondQSwiftEstimator
from gqe.energy_estimator.initializer import XInitializer
from gqe.util import VoidDataset
from qswift.compiler import DefaultOperatorPool
from qwrapper.hamiltonian import HeisenbergModel

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == '__main__':
    nqubit = 3
    N = 8000
    pool = DefaultOperatorPool(all(nqubit))
    # dummy data loader
    dataloader = DataLoader(VoidDataset(), batch_size=1, shuffle=False, num_workers=0)
    # Sampler that samples from generative model
    sampler = NaiveSampler(PauliEnergy(nqubit, 100, gpu=torch.cuda.is_available()),
                           operator_pool=pool, beta=10)
    # Energy estimator
    estimator = SecondQSwiftEstimator(HeisenbergModel(nqubit), XInitializer(),
                                      N, K=0, n_sample=1, n_grad_sample=1)
    # Energy model
    model = EnergyModel(sampler, pool=pool, estimator=estimator, lam=12, nqubit=nqubit, n_samples=100, lr=1e-4).to(
        device)

    recorder = RecordEnergy(sampler, estimator, 100)
    trainer = pl.Trainer(
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        log_every_n_steps=100,
        max_epochs=200,
        gradient_clip_val=None,
        enable_checkpointing=False,
        callbacks=[
            recorder
        ])
    pl.seed_everything(42)
    trainer.fit(model, train_dataloaders=dataloader)
    recorder.save('output/deep_energy.tsv')
