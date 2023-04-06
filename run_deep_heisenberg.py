import random

import torch, pytorch_lightning as pl
import os
from torch.utils.data import DataLoader, Dataset
from gqe.energymodel import PauliEnergy, EnergyModel, NaiveSampler, PrintEnergy
from gqe.estimator import QDriftEstimator
from qwrapper.hamiltonian import HeisenbergModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


class VoidDataset(Dataset):
    def __getitem__(self, index):
        return random.uniform(0, 1)

    def __len__(self):
        return 1


if __name__ == '__main__':
    nqubit = 2
    CHECKPOINT_PATH = "../saved_models/"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    pl.seed_everything(42)

    # dummy data loader
    dataloader = DataLoader(VoidDataset(), batch_size=1,
                            shuffle=False, num_workers=1)

    N = 2000
    sampler = NaiveSampler(PauliEnergy(nqubit), N, lam=6, beta=10, nqubit=nqubit)
    estimator = QDriftEstimator(HeisenbergModel(2), N)
    model = EnergyModel(sampler, estimator=estimator, n_samples=100)

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "deep"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=200,
                         gradient_clip_val=0.1,
                         callbacks=[
                             LearningRateMonitor("epoch"),
                             PrintEnergy(sampler, estimator, 100)
                         ])
    trainer.fit(model, train_dataloaders=dataloader)
