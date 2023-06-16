import random

import torch, pytorch_lightning as pl
from torch.utils.data import DataLoader
from gqe.energy_estimator.general import GeneralEstimator
from gqe.energy_model.general import IIDEnergyModel, all
from gqe.energy_model.callback import V2ExactRecordEnergy
from gqe.energy_model.sampler import V2NaiveSampler
from gqe.energy_model.network import PauliEnergy
from gqe.util import VoidDataset
from qswift.initializer import XBasisInitializer
from qswift.compiler import DefaultOperatorPool
from qwrapper.hamiltonian import HeisenbergModel

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == '__main__':
    nqubit = 3
    N = 1000
    lam = 12
    pool = DefaultOperatorPool(all(nqubit))
    # dummy data loader
    dataloader = DataLoader(VoidDataset(), batch_size=1, shuffle=False, num_workers=0)
    # Sampler that samples from generative model
    sampler = V2NaiveSampler(PauliEnergy(nqubit, 100, gpu=torch.cuda.is_available()),
                             operator_pool=pool, beta=10)
    # Energy estimator
    estimator = GeneralEstimator(pool, HeisenbergModel(nqubit), XBasisInitializer(), tau=lam / N)
    # Energy model
    model = IIDEnergyModel(sampler, pool=pool, estimator=estimator, N=N, lr=1e-4).to(device)
    trainer = pl.Trainer(
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        log_every_n_steps=100,
        max_epochs=200,
        gradient_clip_val=None,
        enable_checkpointing=False,
        callbacks=[
            V2ExactRecordEnergy(sampler, estimator, lam, N, n_samples=100)
        ])
    pl.seed_everything(42)
    trainer.fit(model, train_dataloaders=dataloader)
    # recorder.save('output/deep_energy.tsv')
