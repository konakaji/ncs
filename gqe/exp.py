import sys

import numpy as np
import math
import emcee


# class NaiveSampler(Sampler):
#     def __init__(self, nn, pool: OperatorPool):
#         self.nn = nn
#         self.pool = pool
#
#     def sample_indices(self, count=1):
#
#     def sample_operators(self, count=1) -> [ControllablePauli]:
#         pass
#
#     def sample_time_evolutions(self, count=1) -> [PauliTimeEvolution]:
#         pass
#
#     def get(self, index) -> ControllablePauli:
#         pass
#
#
# class AllPauliObservables(OperatorPool):
#     def __init__(self, nqubit):
#         self.nqubit = nqubit
#         self.cache = self.all()
#
#     def all(self):
#         if self.cache is not None:
#             return self.cache
#         results = [""]
#         for _ in range(self.nqubit):
#             next = []
#             for pauli in ["I", "X", "Y", "Z"]:
#                 for r in results:
#                     r = r + pauli
#                     next.append(r)
#             results = next
#         paulis = []
#         for r in results:
#             paulis.append(PauliObservable(r))
#         for r in results:
#             paulis.append(PauliObservable(r, -1))
#         self.cache = paulis
#         return self.cache
#
#     def get(self, index):
#         return self.cache[index]
#
#     def uniform_sample(self) -> PauliObservable:
#         pass
#
#     def size(self):
#         return len(self.cache)


# Define the probability function
def prob(x):
    for v in x:
        if v > 4 or v < 0:
            return 0.000001
    prob_vals = [0.1, 0.3, 0.2, 0.4]  # discrete probabilities for x in [0, 1, 2, 3]
    v = math.floor(x[0])
    return prob_vals[v]


# Define the log-probability function
def log_prob(x):
    for v in x:
        if v > 4 or v < 0:
            return -sys.maxsize
    res = 0
    for r in [v1 * v2 for v1, v2 in zip(x, x)]:
        res -= r
    return res


# Set up the MCMC sampler
n_dim = 4
n_walkers = 100
n_steps = 100
sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)

# Initialize the walkers
p0 = np.random.uniform(0, 4, size=(n_walkers, n_dim))

# Run the MCMC sampler
sampler.run_mcmc(p0, n_steps)

# Access the results
samples = sampler.get_chain(flat=True)
log_prob_samples = sampler.get_log_prob()
acceptance_fraction = sampler.acceptance_fraction

sampler.run_mcmc(None, n_steps)

samples = sampler.get_chain(flat=True, discard=10)

print([s for s in samples])

# Print some statistics
print("Acceptance fraction:", acceptance_fraction)
print("Mean log-probability:", np.mean(log_prob_samples))

print(len(samples))

# Print the histogram of samples
hist = np.histogram(samples, bins=[0, 1, 2, 3, 4])[0]
print("Histogram of samples:", hist)
