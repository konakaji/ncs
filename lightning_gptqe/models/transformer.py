import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Config
from lightning import LightningModule

class Transformer(LightningModule):
    def __init__(self, cfg, distance):
        super().__init__()
        self._distance = distance
        self.cfg = cfg
        gpt2cfg = GPT2Config(**{ k: cfg[k] for k in GPT2Config().to_dict().keys() & cfg.keys() })
        self.transformer = GPT2LMHeadModel(gpt2cfg)
        self.ngates = cfg.ngates
        self.energy_scaling = cfg.energy_scaling
        self.num_samples = cfg.num_samples
        self.temperature = cfg.temperature
        self.save_hyperparameters()
        self._starting_idx = torch.zeros(cfg.num_samples, 1, dtype=torch.int)
        self.loss_fn = torch.nn.MSELoss()
    
    def generate_logits(self, idx):
        # device = idx.device
        # b, t = idx.size()
        # assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        # forward the GPT model itself
        logits = self.transformer(idx)[0]
        return logits
    
    def set_cost(self, cost):
        self._cost = cost

    def train_step(self):
        log_values = {}
        idx_output, logits_tensor = self.generate()
        energies = self._cost.energy(idx_output)
        mean_logits = torch.mean(logits_tensor, 1)
        log_values[f"mean_logits at {self._distance}"] = torch.mean(mean_logits * self.energy_scaling)
        # log_values[f"energies at {self._distance}"] = energies
        log_values[f"mean energy at {self._distance}"] = torch.mean(energies)
        loss = self.loss_fn(torch.exp(-mean_logits), torch.exp(-energies / self.energy_scaling))
        log_values[f"loss at {self._distance}"] = loss
        return loss, log_values
        # return loss(mean_logits, energies)

    def generate(self, idx=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if idx is None:
            idx = self._starting_idx.clone()
        assert isinstance(idx, torch.IntTensor)
        b_size = idx.shape[0]
        condition_length = idx.size(dim=1)
        # logits_tensor = torch.empty((max_new_tokens, b_size))
        for _ in range(self.ngates):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx
            # idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits_base = self.generate_logits(idx_cond)
            logits = logits_base[:, -1, :]
            probs = F.softmax(-self.temperature * logits, dim=-1)
            # either sample from the distribution or take the most likely element
            idx_next = torch.multinomial(probs, num_samples=1)
            # logits_tensor[pos] = torch.gather(logits, 1, idx_next).flatten()
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        idx = idx[:, condition_length:]
        # print(logits_tensor.T)
        return idx, torch.gather(logits_base, 2, idx.reshape(b_size, -1, 1)).reshape(b_size, -1)

    def forward(self):
        pass