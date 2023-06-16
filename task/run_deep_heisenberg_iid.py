import torch, pytorch_lightning as pl
from torch.utils.data import DataLoader
from gqe.energy_model.iid import IIDEnergyModel
from gqe.energy_model.callback import IIDRecordEnergy
from gqe.energy_model.sampler import NaiveSampler
from gqe.energy_model.network import PauliEnergy
from gqe.energy_estimator.iid import IIDEstimator
from gqe.operator_pool.op import AllPauliOperators
from gqe.util import VoidDataset
from qswift.initializer import XBasisInitializer
from qwrapper.hamiltonian import HeisenbergModel

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == '__main__':
    nqubit = 3
    N = 8000
    n_sample = 1000
    lam = 12
    # dummy data loader
    dataloader = DataLoader(VoidDataset(), batch_size=10, shuffle=False, num_workers=0)
    # operator pool
    pool = AllPauliOperators(nqubit)
    # Sampler that samples from generative model
    sampler = NaiveSampler(PauliEnergy(nqubit, 100, gpu=torch.cuda.is_available()),
                           operator_pool=pool, N=N, lam=lam, beta=10)
    # Energy estimator
    estimator = IIDEstimator(HeisenbergModel(nqubit), XBasisInitializer(),
                             N, K=0, tool='qulacs', n_sample=n_sample, n_grad_sample=2)
    # Energy model
    model = IIDEnergyModel(sampler, estimator=estimator, pool=pool, num_grad=10, lam=lam, lr=1e-4).to(device)

    recorder = IIDRecordEnergy(sampler, estimator, pool, lam)
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
