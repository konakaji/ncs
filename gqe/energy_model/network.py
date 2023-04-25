from abc import ABC, abstractmethod

import torch
from qwrapper.obs import PauliObservable
from torch import nn


class LSTMEnergy(nn.ModuleList):
    def __init__(self, hidden_dim, gpu=False) -> None:
        super().__init__()
        self.encoder = WordEncoder()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(4, self.hidden_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.5)
        self.LSTM_layers = 1
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, 1)
        if gpu:
            self.lstm.to('gpu')
            self.fc1.to('gpu')
            self.fc2.to('gpu')

    def forward(self, paulis: [PauliObservable]):
        h = torch.zeros((self.LSTM_layers, len(paulis), self.hidden_dim))
        c = torch.zeros((self.LSTM_layers, len(paulis), self.hidden_dim))

        # Initialization fo hidden and cell states
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out = self.encoder.encode(paulis)
        out = self.embedding(out)
        out, (hidden, cell) = self.lstm(out, (h, c))
        # out = self.dropout(out)
        # The last hidden state is taken
        out = torch.relu_(self.fc1(out[:, -1, :]))
        # out = self.dropout(out)
        out = self.fc2(out)
        result = torch.zeros(out.shape)
        for j, o in enumerate(out):
            result[j] = o * paulis[j].sign
        return result


class PauliEnergy(nn.ModuleList):
    def __init__(self, nqubit, hidden_dim=100, gpu=False) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoder = SymmetricEncoder()
        self.gpu = gpu
        self.fc1 = nn.Linear(in_features=3 * nqubit, out_features=self.hidden_dim)
        self.fc3 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        if gpu:
            self.fc1 = self.fc1.to('cuda')
            self.fc3 = self.fc3.to('cuda')
            self.fc2 = self.fc2.to('cuda')

    def forward(self, paulis: [PauliObservable]):
        out = self.encoder.encode(paulis)
        out = out.view(out.size(0), -1)
        if self.gpu:
            out = out.to('cuda')
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc3(out)
        out = torch.relu(out)
        out = self.fc2(out)
        result = torch.zeros(out.shape)
        for j, o in enumerate(out):
            result[j] = o * paulis[j].sign
        return result


class Encoder(ABC):
    @abstractmethod
    def encode(self, paulis: [PauliObservable]):
        pass


class WordEncoder(Encoder):
    def encode(self, paulis: [PauliObservable]):
        results = []
        for pauli in paulis:
            result = []
            for c in pauli.p_string:
                chars = ['I', 'X', 'Y', 'Z']
                for j, char in enumerate(chars):
                    if c == char:
                        result.append(j)
            results.append(result)
        return torch.tensor(results, dtype=torch.int32)


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


class SymmetricEncoder(Encoder):
    def encode(self, paulis: [PauliObservable]):
        results = []
        for pauli in paulis:
            result = []
            for c in pauli.p_string:
                r = [0] * 3
                if c == 'I':
                    r[0] = 1
                elif c == 'X':
                    r[1] = 1
                else:
                    r[2] = 1
                result.append(r)
            results.append(result)
        return torch.tensor(results, dtype=torch.float32)