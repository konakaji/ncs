"""
Dataset lists all possible paulis
Batch size lists all possible gates
    -> each batch now corresponds to a complete circuit
"""
import torch
from torch.utils.data import Dataset, DataLoader

class PaulisDataset(Dataset):
    """Implemented for Symmetric Encoder"""
    def __init__(self, npaulis, length):
        self.npaulis = npaulis
        self.length = length

    def __getitem__(self, index):
        # return torch.arange(self.npaulis)
        qubits = torch.tensor([0,1,2,2]) # YZ the same in symmetric encoder
        paulis = torch.cartesian_prod(qubits,qubits,qubits)
        all_paulis = torch.nn.functional.one_hot(paulis,num_classes=3)
        all_paulis = torch.cat([all_paulis, -all_paulis])
        # create mapping back to ixyz
        ixyz = torch.arange(4)
        mapping = torch.cartesian_prod(ixyz,ixyz,ixyz)
        mapping = torch.cat([mapping, mapping])
        return all_paulis.view(all_paulis.shape[0],-1).float(), mapping

    def __len__(self):
        return self.length


def setup_dataloader(cfg):
    # set size of dataset to be multiple of number of gates in circuit
    # i.e., multiplier is the number of batches per epoch
    dataset = PaulisDataset(cfg.npaulis, cfg.ngates*10) 
    dataloader = DataLoader(dataset, batch_size=cfg.ngates)
    return dataloader
