from scipy.linalg import expm

import math
import numpy as np, random
from abc import ABC, abstractmethod
from scipy.linalg import expm

X = np.matrix([[0, 1], [1, 0]])
Y = np.matrix([[0, -1j], [1j, 0]])
Z = np.matrix([[1, 0], [0, -1]])


def zero_state(dim):
    result = np.diag(np.zeros(dim))
    result[0][0] = 1
    return result


class State:
    def __init__(self, hs, matrices):
        self.hs = hs
        self.matrices = matrices

    def get_num_parameters(self):
        return len(self.hs)

    def get_dimension(self):
        return len(self.matrices[0])

    def sum(self):
        result = 0
        for h, m in zip(self.hs, self.matrices):
            if result is None:
                result = (h + 0j) * m
            else:
                result += (h + 0j) * m
        return result

    def lam(self):
        result = 0
        for h in self.hs:
            result += h
        return result


class Channel:
    def __init__(self, m, inv):
        self.m = m
        self.inv = inv

    def apply(self, rho):
        return self.m.dot(rho).dot(self.inv)


class EnergyComputation(ABC):
    def __init__(self, obs, t):
        self.obs = obs
        self.t = t

    @abstractmethod
    def compute_energy(self, state: State):
        pass


class GradientComputation(ABC):
    def __init__(self, obs, t):
        self.obs = obs
        self.t = t

    def compute_grad(self, state: State):
        grads = []
        for p_index in range(state.get_num_parameters()):
            grads.append(self.do_compute_grad(state, p_index))
        return grads

    @abstractmethod
    def do_compute_grad(self, state: State, p_index):
        pass


class ExactEnergyComputation(EnergyComputation):
    def compute_energy(self, state: State):
        channel = Channel(expm(1j * state.sum() * self.t), expm(-1j * state.sum() * self.t))
        rho = channel.apply(zero_state(state.get_dimension()))
        return np.trace(rho.dot(self.obs)).real


class ExactEnergyGradientComputation(GradientComputation):
    def __init__(self, obs, t, delta):
        super().__init__(obs, t)
        self.delta = delta

    def do_compute_grad(self, state, p_index):
        plus_hs = []
        for j, h in enumerate(state.hs):
            if j == p_index:
                plus_hs.append(h + self.delta)
            else:
                plus_hs.append(h)
        plus_state = State(plus_hs, state.matrices)
        ec = ExactEnergyComputation(self.obs, self.t)
        return (ec.compute_energy(plus_state) - ec.compute_energy(state)) / self.delta


class QDriftEnergyComputation(EnergyComputation):
    def __init__(self, obs, N, t):
        super().__init__(obs, t)
        self.N = N

    def compute_energy(self, state: State):
        sampler = ChannelSampler(state.hs, state.matrices)
        rho = zero_state(state.get_dimension())
        for _ in range(self.N):
            index, channel = sampler.sample_channel(tau=self.t)
            rho = channel.apply(rho)
        return np.trace(self.obs.dot(rho)).real


class QDriftUtil:
    @classmethod
    def q(cls, state, p_index):
        return state.hs[p_index] / state.lam()

    @classmethod
    def weight(cls, state):
        return 1 / state.lam()


class QDriftGradientManipulator:
    def __init__(self, grad_layer, deriv_channel):
        self.grad_layer = grad_layer
        self.deriv_channel = deriv_channel

    def manipulate(self, channels):
        channels[self.grad_layer] = self.deriv_channel


class QDriftGradientComputationBase(GradientComputation, ABC):
    def __init__(self, obs, t, N):
        super().__init__(obs, t)
        self.N = N

    def do_compute_grad(self, state: State, deriv_index):
        result = self.first_term(state, deriv_index)
        result += self.second_term(state)
        return result

    def first_term(self, state, deriv_index):
        result = 0
        indices = self.get_layer_indices()
        for grad_layer in indices:
            result += self.layer_grad(state, deriv_index, grad_layer)
        return result

    def second_term(self, state):
        result = 0
        indices = self.get_layer_indices()
        for grad_layer in indices:
            result += self.layer_param_shift(state, grad_layer)
        return result

    def layer_grad(self, state: State, deriv_index, grad_layer):
        normal_sampler = ChannelSampler(state.hs, state.matrices)
        channels = []
        for c in range(self.N):
            if c == grad_layer:
                channels.append(normal_sampler.get_channel(self.t, deriv_index)[1])
            else:
                channels.append(normal_sampler.sample_channel(self.t)[1])
        result = 1 / state.lam() * self.apply(state, channels)
        p_indices = self.get_parameter_indices(state, deriv_index)
        for p_index in p_indices:
            manipulator = QDriftGradientManipulator(grad_layer, normal_sampler.get_channel(self.t, p_index)[1])
            value = self.parameter_grad(state, channels, manipulator, deriv_index, p_index)
            result += value
        return result

    def parameter_grad(self, state: State, channels, manipulator, deriv_index, p_index):
        manipulator.manipulate(channels)
        rho = zero_state(state.get_dimension())
        for c in channels:
            rho = c.apply(rho)
        result = - self.weight(state, deriv_index, p_index) * self.apply(state, channels)
        return result

    def layer_param_shift(self, state, grad_layer):
        normal_sampler = ChannelSampler(state.hs, state.matrices)
        plus_channels = []
        minus_channels = []
        for layer in range(self.N):
            index, channel = normal_sampler.sample_channel(self.t)
            if layer == grad_layer:
                plus_channels.append(normal_sampler.get_channel(self.t + math.pi / 2, index)[1])
                minus_channels.append(normal_sampler.get_channel(self.t - math.pi / 2, index)[1])
            else:
                plus_channels.append(channel)
                minus_channels.append(channel)
        rho_plus = zero_state(state.get_dimension())
        rho_minus = zero_state(state.get_dimension())
        for c in plus_channels:
            rho_plus = c.apply(rho_plus)
        for c in minus_channels:
            rho_minus = c.apply(rho_minus)
        result = self.t / (2 * state.lam()) * (np.trace(self.obs.dot(rho_plus)).real
                                               - np.trace(self.obs.dot(rho_minus)).real)
        return result

    def get_layer_indices(self):
        return [j for j in range(self.N)]

    def apply(self, state, channels):
        rho = zero_state(state.get_dimension())
        for c in channels:
            rho = c.apply(rho)
        return np.trace(self.obs.dot(rho)).real

    @abstractmethod
    def weight(self, state, deriv_index, p_index):
        pass

    @abstractmethod
    def get_parameter_indices(self, state, deriv_index):
        return []


class QDriftExactGradientComputation(QDriftGradientComputationBase):
    def get_parameter_indices(self, state, deriv_index):
        return [p for p in range(state.get_num_parameters())]

    def weight(self, state, deriv_index, p_index):
        weight = QDriftUtil.weight(state)
        q = QDriftUtil.q(state, p_index)
        return weight * q


class QDriftSampleGradientComputation(QDriftGradientComputationBase):
    def __init__(self, obs, t, N, M):
        super().__init__(obs, t, N)
        self.M = M

    def get_parameter_indices(self, state, deriv_index):
        sampler = ImportanceSampler(state.hs)
        results = []
        for _ in range(self.M):
            index = sampler.sample_index()
            results.append(index)
        return results

    def weight(self, state, deriv_index, p_index):
        weight = QDriftUtil.weight(state) / self.M
        return weight


class ImportanceSampler:
    def __init__(self, coeffs):
        self.borders = []
        total = 0
        for coeff in coeffs:
            total += coeff
            self.borders.append(total)
        self.total = total

    def sample_index(self):
        v = random.uniform(0, self.total)
        for j, b in enumerate(self.borders):
            if v < b:
                return j

    def _get_total(self):
        return self.borders[len(self.borders) - 1]


class ChannelSampler:
    def __init__(self, coeffs, operators):
        self.operators = operators
        self.index_sampler = ImportanceSampler(coeffs)

    def sample_channel(self, tau):
        index = self.index_sampler.sample_index()
        return self.get_channel(tau, index)

    def get_channel(self, tau, index):
        m = expm(1j * self.operators[index] * tau / 2)
        inv_m = expm(-1j * self.operators[index] * tau / 2)
        return index, Channel(m, inv_m)


if __name__ == '__main__':
    obs = np.kron(Z, Z)

    num_parameters = 100
    ps = []
    params = []
    for p_index in range(num_parameters):
        ps.append(np.kron(random.choice([X, Y, Z]), random.choice([X, Y, Z])))
        params.append(random.uniform(0, 0.02))

    state = State(params, ps)
    t = 0.5

    exact_computation = ExactEnergyGradientComputation(obs, t, 0.001)
    qdrift_computation = QDriftExactGradientComputation(obs, t * 2 * state.lam() / 10, 10)
    qdrift_another_computation = QDriftSampleGradientComputation(obs, t * 2 * state.lam() / 10, 10, 100)

    grads_1 = exact_computation.compute_grad(state)
    grads_2 = qdrift_computation.compute_grad(state)
    grads_3 = qdrift_another_computation.compute_grad(state)
    #
    print(grads_1)
    print(grads_2)
    print(grads_3)
