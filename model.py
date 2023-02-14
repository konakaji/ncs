import math
import random
from qwrapper.circuit import init_circuit
from qwrapper.obs import PauliObservable
from gqe.sampler import DefaultImportantSampler
import torch
from torch import nn, optim


def to_value(hs):
    rs = [abs(h.item()) for h in hs]
    return rs


def cost(obs: PauliObservable, G, t, N, l_size, n_sample):
    sampler = PGateSampler(G, t / N, l_size)
    res = 0
    for _ in range(n_sample):
        qc = init_circuit(1, "qulacs")
        for _ in range(N):
            sampler.add([qc])
        res += obs.exact_value(qc) / n_sample
    return res


def step(obs: PauliObservable, G, t, N, l_size):
    sampler = PGateSampler(G, t / N, l_size)
    val = 0
    k = random.randint(0, N - 1)
    zs = torch.tensor([random.gauss(0, 1) for _ in range(l_size)])
    hs = G(zs)
    qcs = [init_circuit(1, "qulacs") for _ in range(len(hs))]
    for v in range(N):
        if v == k:
            for index in range(len(hs)):
                PGate.add(qcs[index], index, t / N)
        else:
            sampler.add(qcs)

    results = []
    for index, h in enumerate(hs):
        results.append(obs.exact_value(qcs[index]))
    coeffs = torch.tensor(results)
    c_loss = CustomLoss()
    loss = c_loss(coeffs, hs)
    loss.backward()


class CustomLoss(nn.Module):
    def __init__(self) -> None:
        super(CustomLoss, self).__init__()

    def forward(self, coeffs, hs):
        value = torch.inner(coeffs, hs)
        return value


class PGate:
    @classmethod
    def add(cls, qc, index, time):
        if index == 1:
            qc.rx(time, 0)
        elif index == 2:
            qc.ry(time, 0)
        elif index == 3:
            qc.rz(time, 0)


class PGateSampler:
    def __init__(self, generator, time, l_size):
        self.generator = generator
        self.time = time
        self.l_size = l_size

    def add(self, qcs):
        zs = torch.tensor([random.gauss(0, 1) for _ in range(self.l_size)])
        hs = G(zs)
        sampler = DefaultImportantSampler(to_value(hs))
        index = sampler.sample_index()
        for label, qc in enumerate(qcs):
            PGate.add(qc, index, self.time)


class ProbGenerator(nn.Module):
    def __init__(self, latent_size,
                 hidden_size, output_size):
        super(ProbGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.soft_max = nn.Softmax(0)

    def forward(self, z):
        x = self.network.forward(z)
        x = self.soft_max(x)
        return x


if __name__ == '__main__':
    l_size = 10
    h_size = 200
    o_size = 4
    N = 200
    t = 3 * math.pi
    n_step = 100
    n_sample = 100

    G = ProbGenerator(l_size, h_size, o_size)
    optimizer = optim.SGD(G.parameters(), lr=0.001, momentum=0.5)
    for j in range(n_step):
        for _ in range(n_sample):
            optimizer.zero_grad()
            step(PauliObservable("X"), G, t, N, l_size)
            optimizer.step()
        print(j, cost(PauliObservable("X"), G, t, N, l_size, 100))
