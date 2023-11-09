import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Config
from lightning import LightningModule
from gqe.util import get_device


class Transformer(LightningModule):
    def __init__(self, cfg, label):
        super().__init__()
        self._label = label
        self.cfg = cfg
        gpt2cfg = GPT2Config(**{k: cfg[k] for k in GPT2Config().to_dict().keys() & cfg.keys()})
        self.transformer = GPT2LMHeadModel(gpt2cfg).to(get_device())
        self.ngates = cfg.ngates
        self.energy_offset = cfg.energy_offset
        self.num_samples = cfg.num_samples
        self.temperature = cfg.temperature
        self.save_hyperparameters()
        self._starting_idx = torch.zeros(cfg.num_samples, 1, dtype=torch.int, device=get_device())
        self.loss_fn = torch.nn.MSELoss()

    def generate_logits(self, idx):
        logits = self.transformer(idx)[0]
        return logits

    def set_cost(self, cost):
        self._cost = cost

    def gather(self, idx, logits_base):
        b_size = idx.shape[0]
        return torch.gather(logits_base, 2, idx.reshape(b_size, -1, 1)).reshape(b_size, -1)

    def train_step(self, indices=None, energies=None):
        log_values = {}
        if energies is not None:
            assert indices is not None
            idx_output = indices[:, 1:]
            logits_base = self.generate_logits(idx_output)
        else:
            idx_output, logits_base = self.generate()
            energies = self._cost.energy(idx_output)
        logits_tensor = self.gather(idx_output, logits_base)
        mean_logits = torch.mean(logits_tensor, 1)
        log_values[f"mean_logits at {self._label}"] = torch.mean(mean_logits - self.energy_offset)
        log_values[f"mean energy at {self._label}"] = torch.mean(energies)
        loss = self.loss_fn(torch.exp(-mean_logits), torch.exp(-energies - self.energy_offset))
        log_values[f"loss at {self._label}"] = loss
        return loss, energies, idx_output, log_values

    def generate(self, idx=None, ngates=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if idx is None:
            idx = self._starting_idx.clone()
        condition_length = idx.size(dim=1)
        if ngates is None:
            ngates = self.ngates
        for _ in range(ngates):
            idx_cond = idx
            logits_base = self.generate_logits(idx_cond)
            logits = logits_base[:, -1, :]
            probs = F.softmax(-self.temperature * logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        idx = idx[:, condition_length:]
        return idx, logits_base

    def forward(self):
        self.train_step()
