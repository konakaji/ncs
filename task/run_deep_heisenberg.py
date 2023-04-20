import random

import torch, pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from gqe.energy_model.model import EnergyModel
from gqe.energy_model.callback import RecordEnergy
from gqe.energy_model.sampler import NaiveSampler
from gqe.energy_model.network import PauliEnergy
from gqe.energy_estimator.qdrift import QDriftEstimator
from qwrapper.hamiltonian import HeisenbergModel

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class VoidDataset(Dataset):
    def __init__(self, count=10):
        self.count = count

    def __getitem__(self, index):
        return [random.uniform(0, 1) for _ in range(self.count)]

    def __len__(self):
        return self.count


if __name__ == '__main__':
    nqubit = 3
    N = 8000

    # dummy data loader
    dataloader = DataLoader(VoidDataset(), batch_size=1, shuffle=False, num_workers=0)
    # Sampler that samples from generative model
    sampler = NaiveSampler(PauliEnergy(nqubit, 100, gpu=torch.cuda.is_available()), N, lam=12, beta=10, nqubit=nqubit)
    # Energy estimator
    estimator = QDriftEstimator(HeisenbergModel(nqubit), N, tool='qulacs')
    # Energy model
    model = EnergyModel(sampler, estimator=estimator, n_samples=100, lr=1e-4).to(device)

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
