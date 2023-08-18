import logging

import torch, pytorch_lightning as pl
from torch.utils.data import DataLoader
from gqe.energy_model.iid import IIDEnergyModel
from gqe.energy_model.callback import RecordEnergy
from gqe.energy_model.sampler import NaiveSampler
from gqe.energy_model.network import PauliEnergy
from gqe.energy_estimator.qdrift import QDriftEstimator
from gqe.hamiltonian.molecule import MolecularHamiltonian
from gqe.measurement import StochasticMeasurementMethod, AncillaStochasticMeasurementMethod
from gqe.operator_pool.depr_uccsd import UCCSD
from gqe.util import VoidDataset

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == '__main__':
    ##################################
    ### Basic settings ###############
    ##################################
    nqubit = 12
    N = 800
    lam = 12
    hidden_dim = 100
    sampler = NaiveSampler(PauliEnergy(nqubit, hidden_dim=hidden_dim, gpu=torch.cuda.is_available()),
                           operator_pool=UCCSD(nqubit),
                           N=N, lam=lam, beta=10)
    hamiltonian = MolecularHamiltonian(nqubit, "sto-3g", pubchem_name="lih")
    ##################################

    # Energy estimator
    estimator = QDriftEstimator(hamiltonian, N, measurement=StochasticMeasurementMethod(hamiltonian, 1),
                                ancilla_measurement=AncillaStochasticMeasurementMethod(hamiltonian, 1), tool='qulacs')
    # Energy model
    model = IIDEnergyModel(sampler, estimator=estimator, n_samples=100, lr=1e-4).to(device)

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
    logging.getLogger().setLevel(logging.INFO)
    # dummy data loader
    dataloader = DataLoader(VoidDataset(), batch_size=1, shuffle=False, num_workers=0)
    trainer.fit(model, train_dataloaders=dataloader)
    recorder.save('output/deep_energy.tsv')
