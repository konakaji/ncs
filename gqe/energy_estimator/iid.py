from qwrapper.obs import Hamiltonian
from qwrapper.sampler import ImportantSampler
from qwrapper.hamiltonian import to_matrix_hamiltonian
from qswift.util import make_positive
from qswift.measurement import NaiveGenerator
from qswift.compiler import Compiler, SwiftChannel, OperatorPool, MultiIndexSampler
from qswift.executor import QSwiftExecutor
from qswift.qswift import QSwift
from qswift.exact import ExactComputation
import random, sys, logging, time, math


class IIDEstimator:
    def __init__(self, obs: Hamiltonian, initializer, N, K, n_sample, n_grad_sample, tool='qulacs', shot=0, executor=None):
        """
        :param obs: Hamiltonian, where we want to get the ground state
        :param initializer: The initializer of the quantum circuit
        :param N: # of quantum gates in each circuit
        :param K: The order of the energy estimation
        :param n_sample: # of samples to estimate the energy
        :param n_grad_sample: # of samples to estimate the gradient
        :param tool: qiskit or qulacs
        :param shot: # of shots for each sample
        """
        self.logger = logging.getLogger("gqe.energy_estimator.qswift.IIDEstimator")
        if K > 1:
            raise AttributeError("K > 1 is not implemented")
        obs = make_positive(obs)
        self.hamiltonian = obs
        self.tool = tool
        self.qswift = QSwift(obs, initializer, t=1, N=N, K=K, n_p=n_sample,tool=self.tool)
        self.obs = obs.gen_ancilla_hamiltonian("X")
        if shot == 0:
            # Directly calculate the expectation value of the Hamiltonian
            self.measurement_gen = NaiveGenerator([1])
        else:
            # Generate measurements for each pauli in the Hamiltonian one by one
            self.measurement_gen = NaiveGenerator(obs.hs)
        self.initializer = initializer
        self.executor = QSwiftExecutor() if executor == None else executor
        self.N = N
        self.K = K
        self.tool = tool 
        self.shot = shot
        self.n_sample = n_sample
        self.n_grad_sample = n_grad_sample
        # Order of the gradient calculation
        self.grad_order = 3

    def exact(self, evolution_hamiltonian: Hamiltonian):
        """
        :return: Tr(H e^{i\sum_{j=1}h_j O_j} rho e^{-i\sum_{j=1}h_j O_j})}
        """
        computation = ExactComputation(to_matrix_hamiltonian(self.hamiltonian),
                                       to_matrix_hamiltonian(evolution_hamiltonian), 1, self.initializer)
        return computation.compute()

    def value(self, sampler: ImportantSampler, operator_pool: OperatorPool, lam):
        """
        :param sampler:
        :param operator_pool:
        :param lam:
        :return : The value of the current energy estimated by qSWIFT
        """
        start = time.time()
        result = self.qswift.evaluate(sampler=sampler, operator_pool=operator_pool, lam=lam).sum(self.K)
        self.logger.debug(f"value: {time.time() - start}")
        return result

    def grad(self, sampler: ImportantSampler, operator_pool: OperatorPool, lam, j):
        compiler = self._build_compiler(operator_pool, lam)
        g = self.executor.execute(compiler, self.g(sampler, lam, j))
        d1 = lam ** 2 / (2 * self.N) * self.executor.execute(compiler, self.d1(sampler, lam, j))
        d2 = lam ** 2 / (2 * self.N) * self.executor.execute(compiler, self.d2(sampler, lam, j))
        self.logger.debug(f"(g, d1, d2) = ({g}, {d1}, {d2})")
        return g + d1 + d2

    def g(self, sampler: ImportantSampler, lam, j):
        channels = []
        for measurement in self.measurement_gen.generate(self.n_grad_sample):
            seed = random.randint(0, sys.maxsize)
            for n in range(1, self.grad_order):
                coeff = 1 / math.factorial(n) * (lam / self.N) ** (n - 1)
                j_vec = sampler.sample_indices(self.N - 1)
                swift_channel = SwiftChannel(coeff)
                swift_channel.add_time_operators(j_vec)
                swift_channel.add_multi_l_operators([j for _ in range(n)])
                swift_channel.shuffle(seed)
                measurement.assign(swift_channel)
                channels.append(swift_channel)
        return channels

    def d1(self, sampler: ImportantSampler, lam, j):
        if self.K == 0:
            return []
        channels = []
        for measurement in self.measurement_gen.generate(self.n_grad_sample):
            seed = random.randint(0, sys.maxsize)
            j_vec = sampler.sample_indices(self.N - 1)
            ell = sampler.sample_index()
            swift_channel = SwiftChannel(1 / lam)
            swift_channel.add_time_operators(j_vec)
            swift_channel.add_multi_l_operators([j, ell])
            swift_channel.shuffle(seed)
            measurement.assign(swift_channel)
            channels.append(swift_channel)

            swift_channel = SwiftChannel(1 / lam)
            swift_channel.add_time_operators(j_vec)
            swift_channel.add_multi_l_operators([ell, j])
            swift_channel.shuffle(seed)
            measurement.assign(swift_channel)
            channels.append(swift_channel)

            swift_channel = SwiftChannel(-1 / lam)
            swift_channel.add_time_operators(j_vec)
            swift_channel.add_multi_l_operators([j, j])
            swift_channel.shuffle(seed)
            measurement.assign(swift_channel)
            channels.append(swift_channel)
        return channels

    def d2(self, sampler: ImportantSampler, lam, j):
        if self.K == 0:
            return []
        multi_sampler = MultiIndexSampler(sampler)
        channels = []
        for measurement in self.measurement_gen.generate(self.n_grad_sample):
            for n in range(1, self.grad_order):
                seed = random.randint(0, sys.maxsize)
                j_vec = sampler.sample_indices(self.N - 2)
                coeff = 1 / math.factorial(n) * (lam / self.N) ** (n - 1)
                for s in [0, 1]:
                    sign = (-1) ** s
                    l1, l2 = multi_sampler.sample(s, 2)
                    swift_channel = SwiftChannel(sign * coeff)
                    swift_channel.add_time_operators(j_vec)
                    swift_channel.add_multi_l_operators([l1, l2])
                    swift_channel.add_multi_l_operators([j for _ in range(n)])
                    swift_channel.shuffle(seed)
                    measurement.assign(swift_channel)
                    channels.append(swift_channel)
        return channels

    def _build_compiler(self, operator_pool: OperatorPool, lam):
        tau = lam / self.N
        if self.shot == 0:
            # Directly calculate the expectation value of the Hamiltonian
            return Compiler(operator_pool=operator_pool,
                            observables=[self.obs],
                            initializer=self.initializer, nshot=self.shot, tau=tau, tool=self.tool)
        else:
            # Evaluate each pauli in the Hamiltonian one by one
            return Compiler(operator_pool=operator_pool,
                            observables=self.obs.paulis,
                            initializer=self.initializer, nshot=self.shot, tau=tau)
