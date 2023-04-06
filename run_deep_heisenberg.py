import random

import torch, pytorch_lightning as pl
import os
from torch.utils.data import DataLoader, Dataset
from gqe.energymodel import PauliEnergy, DeepMCMCSampler, EnergyModel
from gqe.estimator import QDriftEstimator
from qwrapper.hamiltonian import HeisenbergModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from tutorial.callback import OutlierCallback, SamplerCallback, GenerateCallback


class VoidDataset(Dataset):
    def __getitem__(self, index):
        return random.uniform(0, 1)

    def __len__(self):
        return 1


if __name__ == '__main__':
    nqubit = 2
    CHECKPOINT_PATH = "../saved_models/tutorial8"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "MNIST"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=200,
                         gradient_clip_val=0.1,
                         callbacks=[
                             ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
                             GenerateCallback(every_n_epochs=5),
                             SamplerCallback(every_n_epochs=5),
                             OutlierCallback(),
                             LearningRateMonitor("epoch")
                         ])
    pl.seed_everything(42)

    dataloader = DataLoader(VoidDataset(), batch_size=1,
                            shuffle=False, num_workers=1)

    N = 2000
    model = EnergyModel(network=PauliEnergy(nqubit),
                        estimator=QDriftEstimator(HeisenbergModel(2), N), beta=10, N=N, lam=6,
                        n_qubit=nqubit,
                        n_samples=100)
    trainer.fit(model, train_dataloaders=dataloader)
