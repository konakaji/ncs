import random
import torch
from qwrapper.circuit import QWrapper
from qwrapper.sampler import ImportantSampler
from gqe.estimator import EnergyEstimator


class Initializer:
    def initialize(self, qc: QWrapper, targets) -> QWrapper:
        pass


class DeepSampler(ImportantSampler):
    def __init__(self, nn):
        self.nn = nn

    def sample_index(self):
        pass

    def sample_indices(self, count):
        return super().sample_indices(count)


class EnergyModel:
    def __init__(self, network: torch.nn.Module, estimator: EnergyEstimator,
                 beta, N, lam, n_samples):
        self.network = network
        self.estimator = estimator
        self.beta = beta
        self.N = N
        self.lam = lam
        self.n_samples = n_samples
        self.sampler = DeepSampler(self.network)

    def training_step(self):
        indices = self.sampler.sample_indices(self.n_samples)
        fs = self.network.forward(indices)
        mean = fs.mean()
        grads = torch.FloatTensor(len(indices))
        for j, index in enumerate(indices):
            pos = random.randint(0, self.N - 1)
            grads[j] = self.estimator.grad(pos, index)
        loss = torch.dot(fs - mean, grads)
        return loss
