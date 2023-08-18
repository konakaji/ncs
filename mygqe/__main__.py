# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pytorch_lightning as pl
from mygqe.model import EnergyModel
# from mygqe.qdrift import QDriftEstimator
from mygqe.iid import IIDEstimator
from qswift.initializer import XBasisInitializer
from qwrapper.hamiltonian import HeisenbergModel
from experiments.deep_heisenberg import cfg, dataloader


if __name__ == '__main__':
    # Energy estimator
    estimator = IIDEstimator(HeisenbergModel(cfg.nqubit), XBasisInitializer(),
                             cfg.ngates, K=0, tool='qulacs', n_grad_sample=2)
    # estimator = QDriftEstimator(HeisenbergModel(cfg.nqubit), cfg.ngates, tool='qulacs')
    pl_module = EnergyModel(cfg, estimator)

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        log_every_n_steps=100,
        max_epochs=200,
        gradient_clip_val=None,
        enable_checkpointing=False,
        )
    pl.seed_everything(42)
    trainer.fit(pl_module, train_dataloaders=dataloader)
    # recorder.save('output/deep_energy.tsv')
