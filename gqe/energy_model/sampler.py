import math
import random
import sys

import emcee
import numpy as np
from qwrapper.operator import ControllablePauli, PauliTimeEvolution
from qwrapper.sampler import FasterImportantSampler

from gqe.energy_estimator.ee import Sampler


class NaiveSampler(Sampler):
    def __init__(self, nn, N, lam, beta, nqubit):
        self.nn = nn
        self.nqubit = nqubit
        self.all_paulis = self._all_paulis()
        self.N = N
        self.lam = lam
        self.beta = beta
        self.sampler = self.reset()
        self.evolution_map = {}

    def sample_indices(self, count=1):
        return self.sampler.sample_indices(count)

    def sample_operators(self, count=1) -> [ControllablePauli]:
        paulis = []
        for index in self.sampler.sample_indices(count):
            paulis.append(self.all_paulis[index])
        return paulis

    def sample_time_evolutions(self, count=1) -> [PauliTimeEvolution]:
        results = []
        indices = self.sample_indices(count)
        for key in indices:
            if key not in self.evolution_map:
                self.evolution_map[key] = PauliTimeEvolution(self.all_paulis[key], self.lam / self.N)
            results.append(self.evolution_map[key])
        return results

    def get(self, index) -> ControllablePauli:
        return self.all_paulis[index]

    def get_all(self, indices) -> [ControllablePauli]:
        results = []
        for index in indices:
            results.append(self.get(index))
        return results

    def reset(self):
        probs = self._all_probabilities()
        self.sampler = FasterImportantSampler(probs)
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