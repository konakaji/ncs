from .setup_dataloader import setup_dataloader
from argparse import Namespace

cfg = Namespace(**dict(
    nqubit = 3,
    npaulis = 128,
    ngates = 8000,
    lr = 1e-4,
    beta1 = 0.0,
    beta = 10,
    lam = 12,
    hidden_size = 9, 
    attention_hidden_size = 8,  
    num_heads = 4,
    num_gates = 8000,
    num_paulis = 128,
))

dataloader = setup_dataloader(cfg)


__all__ = ['cfg', 'dataloader']
