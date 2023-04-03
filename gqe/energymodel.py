from abc import ABC, abstractmethod

import torch
from torch import nn
from qwrapper.operator import ControllablePauli, PauliTimeEvolution, PauliObservable
from gqe.estimator import EnergyEstimator, Sampler


class PauliEnergy(nn.ModuleList):
    def __init__(self) -> None:
        super().__init__()

        self.hidden_dim = 20
        self.encoder = OneHotEncoder()
        self.LSTM_layers = 2
        self.input_size = 6

        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, 1)

    def forward(self, paulis: [PauliObservable]):
        # Hidden and cell state definion
        h = torch.zeros((self.LSTM_layers, len(paulis), self.hidden_dim))
        c = torch.zeros((self.LSTM_layers, len(paulis), self.hidden_dim))

        # Initialization fo hidden and cell states
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out = self.encoder.encode(paulis)
        out = self.embedding(out)
        out, (hidden, cell) = self.lstm(out, (h, c))
        out = self.dropout(out)
        # The last hidden state is taken
        out = torch.relu_(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class OperatorPool(ABC):
    @abstractmethod
    def uniform_sample(self) -> PauliObservable:
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
            if pauli.sign == 1:
                result.append(4)
            else:
                result.append(5)
            for c in pauli.p_string:
                chars = ['I', 'X', 'Y', 'Z']
                for j, char in enumerate(chars):
                    if c == char:
                        result.append(j)
            results.append(result)
        return torch.tensor(results, dtype=torch.int32)


class DeepMCMCSampler(Sampler):
    def __init__(self, nn, pool: OperatorPool):
        self.nn = nn
        self.pool = pool

    def sample_indices(self, count=1):
        return []

    def sample_operators(self, count=1) -> [ControllablePauli]:
        pass

    def sample_time_evolutions(self, count=1) -> [PauliTimeEvolution]:
        pass

    def get(self, index) -> ControllablePauli:
        pass


class EnergyModel:
    def __init__(self, network: torch.nn.ModuleList, estimator: EnergyEstimator,
                 pool: OperatorPool,
                 beta, N, lam, n_samples):
        self.network = network
        self.estimator = estimator
        self.beta = beta
        self.N = N
        self.lam = lam
        self.n_samples = n_samples
        self.sampler = DeepMCMCSampler(self.network, pool)

    def training_step(self):
        indices = self.sampler.sample_indices(self.n_samples)
        fs = self.network.forward(indices)
        mean = fs.mean()
        grads = torch.FloatTensor(len(indices))
        for j, index in enumerate(indices):
            grads[j] = self.estimator.grad(self.sampler, index)
        loss = -self.lam * self.beta * torch.dot(fs - mean, grads)
        return loss
