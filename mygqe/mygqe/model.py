import torch
import pytorch_lightning as pl
from mygqe.network import PauliEnergy
from qwrapper.operator import ControllablePauli


class EnergyModel(pl.LightningModule):
    def __init__(self, cfg, estimator):
        super().__init__()
        self.cfg = cfg
        self.network = PauliEnergy(
            cfg.hidden_size,
            cfg.attention_hidden_size,
            cfg.num_heads,
            cfg.num_gates,
            cfg.num_paulis,
        )
        self.lam = cfg.lam
        self.estimator = estimator
        self.save_hyperparameters()


    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)  # Exponential decay over epochs
        return [optimizer], [scheduler]


    # def on_train_epoch_end(self) -> None:
    #     energy = self.estimator.value(self.sampler, self.pool, self.lam)
    #     self.records.append(energy)
    #     print(energy)
    #     return super().on_train_epoch_end()

    def training_step(self, batch):
        # circuit shape (8000, 128, 9)
        circuit, mapping = batch
        operator_pool = OperatorPool(mapping[0])
        model_output = self.network(circuit.permute(1,0,2)) # (128, 8000, 1)
        model_output = model_output.squeeze().T # (8000, 128)
        probs = torch.exp( -self.cfg.beta * model_output )
        avg_sampler = torch.distributions.Categorical(probs.mean(axis=0).squeeze())
        indices = avg_sampler.sample([10])
        sampler = torch.distributions.Categorical(probs)
        wrapped_sampler = SamplerWrapper(avg_sampler, sampler, model_output.shape[0])
        gs = [self.estimator.grad(wrapped_sampler, operator_pool, self.lam, index) for index in indices]
        fs = model_output[:,indices].mean(axis=0)
        mean = fs.mean()
        loss = -torch.dot(torch.flatten(fs) - mean, torch.tensor(gs, dtype=torch.float32)) / len(indices)
        

        # rv_discrete(values=(self.indices, self.p)).rvs(size=count)
        
        # gs, indices = self.estimator.grads(self.sampler)
        # fs = self.sampler.nn.forward(self.sampler.get_all(indices))
        # mean = fs.mean()
        # value = -torch.dot(torch.flatten(fs) - mean, torch.tensor(gs, dtype=torch.float32)) / len(indices)

        return loss


class OperatorPool:
    def __init__(self, mapping) -> None:
        self.mapping = mapping

    def get(self, idx):
            pauli_enc = self.mapping[idx]
            sign = -1 if idx >= len(self.mapping)//2 else 1
            pauli_str = ''.join(map(lambda x: {0:"I",1:"X",2:"Y",3:"Z"}[x], pauli_enc.tolist()))
            return ControllablePauli(pauli_str, sign)


class SamplerWrapper:
    def __init__(self, avg_sampler, sampler, ngates) -> None:
        self.N = ngates
        self.avg_sampler = avg_sampler
        self.sampler = sampler
        
    def sample_index(self):
        return self.avg_sampler.sample().item()
    
    def sample_indices(self, n):
        n_samples_to_remove = self.N-n
        indices = self.sampler.sample()
        mask = ~torch.isin(torch.arange(self.N), torch.randint(0,self.N,(n_samples_to_remove,)))
        return indices[mask].tolist()
