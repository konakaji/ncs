import random
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.optim as optim
from torch import nn
from qwrapper.sampler import DefaultImportantSampler
import pytorch_lightning as pl
from qwrapper.operator import ControllablePauli, PauliTimeEvolution, PauliObservable
from gqe.estimator import EnergyEstimator, Sampler, QDriftEstimator
import emcee, sys, math, numpy as np


class PauliEnergy(nn.ModuleList):
    def __init__(self, nqubit, hidden_dim=100) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoder = OneHotEncoder()
        # self.LSTM_layers = 1
        self.fc1 = nn.Linear(in_features=4 * nqubit, out_features=self.hidden_dim)
        self.fc3 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, paulis: [PauliObservable]):
        # Hidden and cell state definion
        # h = torch.zeros((self.LSTM_layers, len(paulis), self.hidden_dim))
        # c = torch.zeros((self.LSTM_layers, len(paulis), self.hidden_dim))
        # Initialization fo hidden and cell states
        # torch.nn.init.xavier_normal_(h)
        # torch.nn.init.xavier_normal_(c)
        # out = self.embedding(out)
        # out, (hidden, cell) = self.lstm(out, (h, c))
        # The last hidden state is taken
        # out = torch.relu_(self.fc1(out[:, -1, :]))
        out = self.encoder.encode(paulis)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc3(out)
        out = torch.relu(out)
        out = self.fc2(out)
        result = torch.zeros(out.shape)
        for j, o in enumerate(out):
            result[j] = o * paulis[j].sign
        return result


class OperatorPool(ABC):
    @abstractmethod
    def all(self):
        pass

    @abstractmethod
    def size(self):
        pass


class Encoder(ABC):
    @abstractmethod
    def encode(self, paulis: [PauliObservable]):
        pass


class OneHotEncoder(Encoder):
    def encode(self, paulis: [PauliObservable]):
        results = []
        for pauli in paulis:
            result = []
            for c in pauli.p_string:
                r = [0] * 4
                chars = ['I', 'X', 'Y', 'Z']
                for j, char in enumerate(chars):
                    if c == char:
                        r[j] = 1
                result.append(r)
            results.append(result)
        return torch.tensor(results, dtype=torch.float32)


class NaiveSampler(Sampler):
    def __init__(self, nn, N, lam, beta, nqubit):
        self.nn = nn
        self.nqubit = nqubit
        self.all_paulis = self._all_paulis()
        self.N = N
        self.lam = lam
        self.beta = beta
        self.sampler = self.reset()

    def sample_indices(self, count=1):
        keys = []
        for p in self.sample_operators(count):
            keys.append(str(p))
        return keys

    def sample_operators(self, count=1) -> [ControllablePauli]:
        paulis = []
        for index in self.sampler.sample_indices(count):
            paulis.append(self.all_paulis[index])
        return paulis

    def sample_time_evolutions(self, count=1) -> [PauliTimeEvolution]:
        return [PauliTimeEvolution(o, self.lam / self.N) for o in self.sample_operators(count)]

    def get(self, index) -> ControllablePauli:
        return ControllablePauli.from_str(index)

    def get_all(self, indices) -> [ControllablePauli]:
        results = []
        for index in indices:
            results.append(self.get(index))
        return results

    def reset(self):
        probs = self._all_probabilities()
        self.sampler = DefaultImportantSampler(probs)
        return self.sampler

    def _all_probabilities(self):
        results = []
        for f in self.nn.forward(self.all_paulis):
            results.append(math.exp(-self.beta * f))
        return results

    def _all_paulis(self):
        results = [""]
        for _ in range(self.nqubit):
            next = []
            for pauli in ["I", "X", "Y", "Z"]:
                for r in results:
                    r = r + pauli
                    next.append(r)
            results = next
        paulis = []
        for r in results:
            paulis.append(ControllablePauli(r))
        for r in results:
            paulis.append(ControllablePauli(r, -1))
        return paulis


class DeepMCMCSampler(Sampler):
    def __init__(self, nn, N, lam, beta, nqubit):
        self.nn = nn
        self.nqubit = nqubit

        # Set up the MCMC sampler
        self.n_dim = self.nqubit
        self.n_walkers = 100
        self.tau = lam / N
        self.beta = beta
        self.current_step = 0
        self.sampler = self._warmup()
        self.cache = []

    def reset(self):
        self.current_step = 0
        self.sampler = self._warmup()

    def sample_indices(self, count=1):
        return [str(p) for p in self.sample_operators(count)]

    def sample_operators(self, count=1) -> [ControllablePauli]:
        n_step = math.ceil(count / self.n_walkers)
        self.sampler.run_mcmc(None, n_step)
        samples = self.sampler.get_chain(flat=True, discard=self.current_step)
        self.current_step += n_step
        results = []
        for s in samples:
            p = self._to_pauli(s)
            results.append(p)
        return results[:count]

    def sample_time_evolutions(self, count=1) -> [PauliTimeEvolution]:
        return [PauliTimeEvolution(o, self.tau) for o in self.sample_operators(count)]

    def get(self, key) -> ControllablePauli:
        return ControllablePauli.from_str(key)

    def get_all(self, indices) -> [ControllablePauli]:
        results = []
        for index in indices:
            results.append(self.get(index))
        return results

    def log_prob(self, vec):
        if vec[self.n_dim] > 1 or -1 > vec[self.n_dim]:
            return -sys.maxsize
        for v in vec[:self.n_dim]:
            if v < 0 or 4 < v:
                return -sys.maxsize
        pauli = self._to_pauli(vec)
        value = -self.beta * self.nn.forward([pauli])
        return value.item()

    def _warmup(self):
        sampler = emcee.EnsembleSampler(self.n_walkers, self.n_dim + 1, self.log_prob)
        p0 = np.random.uniform(0, 4, size=(self.n_walkers, self.n_dim + 1))
        for v in p0:
            v[self.n_dim] = random.uniform(-1, 1)
        sampler.run_mcmc(p0, 10)
        self.current_step += 10
        return sampler

    def _to_pauli(self, vec):
        if vec[self.n_dim] > 0:
            sign = 1
        else:
            sign = -1
        vec = [math.floor(v) for v in vec[:self.n_dim]]
        pstring = ''
        for index in vec:
            if index == 0:
                char = 'I'
            elif index == 1:
                char = 'X'
            elif index == 2:
                char = 'Y'
            elif index == 3:
                char = 'Z'
            else:
                raise AttributeError('invalid index')
            pstring += char
        result = ControllablePauli(pstring, sign)
        return result


class EnergyModel(pl.LightningModule):
    def __init__(self, network: torch.nn.ModuleList, estimator: EnergyEstimator,
                 beta, N, lam, n_qubit, n_samples, lr=1e-4, beta1=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.network = network
        self.estimator = estimator
        self.beta = beta
        self.N = N
        self.lam = lam
        self.n_samples = n_samples
        self.sampler = NaiveSampler(self.network, N, lam, beta, n_qubit)
        # self.sampler = DeepMCMCSampler(self.network, N, lam, beta, n_qubit)

    def forward(self, paulis) -> Any:
        return self.network.forward(paulis)

    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)  # Exponential decay over epochs
        return [optimizer], [scheduler]

    def training_step(self, batch):
        self.sampler.reset()
        indices = self.sampler.sample_indices(self.n_samples)
        unique, counts = np.unique(indices, return_counts=True)
        print(dict(zip(unique, counts)))
        fs = self.network.forward(self.sampler.get_all(indices))
        mean = fs.mean()
        gs = []
        for index in indices:
            gs.append(self.estimator.grad(self.sampler, index))
        loss = -self.lam * self.beta * torch.dot(torch.flatten(fs) - mean, torch.tensor(gs, dtype=torch.float32)) / len(
            indices)
        values = []
        print(loss)
        for j in range(self.n_samples):
            values.append(self.estimator.value(self.sampler))
        print(np.mean(values))
        return loss
