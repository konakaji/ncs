from qwrapper.obs import Hamiltonian
from qwrapper.sampler import ImportantSampler
from qswift.util import make_positive
from qswift.measurement import NaiveGenerator
from qswift.compiler import Compiler, SwiftChannel, OperatorPool
from qswift.executor import QSwiftExecutor
from qswift.qswift import QSwift
import random, sys, logging, time


class SecondQSwiftEstimator:
    def __init__(self, obs: Hamiltonian, initializer, N, n_sample, n_grad_sample, tool='qulacs', shot=0):
        self.logger = logging.getLogger("gqe.energy_estimator.qswift.SecondQSwiftEstimator")
        obs = make_positive(obs)
        self.obs = obs
        self.measurement_gen = NaiveGenerator(obs.hs)
        self.qswift = QSwift(obs, initializer, t=1, N=N, K=0, n_p=n_sample)
        self.initializer = initializer
        self.executor = QSwiftExecutor()
        self.N = N
        self.tool = tool
        self.shot = shot
        self.n_sample = n_sample
        self.n_grad_sample = n_grad_sample

    def value(self, sampler: ImportantSampler, operator_pool: OperatorPool, lam):
        start = time.time()
        result = self.qswift.evaluate(sampler=sampler, operator_pool=operator_pool, lam=lam).sum(0)
        self.logger.debug(f"value: {time.time() - start}")
        return result

    def grad(self, sampler: ImportantSampler, operator_pool: OperatorPool, lam, j):
        compiler = self._build_compiler(operator_pool, lam)
        channels = []
        channels.extend(self.first_order_grad(sampler, j))
        channels.extend(self.second_order_grad(sampler, lam, j))
        self.executor.execute(compiler, channels)

    def first_order_grad(self, sampler: ImportantSampler, j):
        channels = []
        for measurement in self.measurement_gen.generate(self.n_grad_sample):
            seed = random.randint(0, sys.maxsize)
            j_vec = sampler.sample_indices(self.N - 1)
            for b in [0, 1]:
                swift_channel = SwiftChannel(1 / 2)
                swift_channel.add_time_operators(j_vec)
                swift_channel.add_swift_operator(j, b)
                swift_channel.shuffle(seed)
                measurement.assign(swift_channel)
                channels.append(swift_channel)
        return channels

    def second_order_grad(self, sampler: ImportantSampler, lam, j):
        channels = []
        coeff = lam ** 2 / (2 * self.N)
        for measurement in self.measurement_gen.generate(self.n_grad_sample):
            j_vec = sampler.sample_indices(self.N - 1)
            seed = random.randint(0, sys.maxsize)
            for b in [0, 1]:
                swift_channel = SwiftChannel(2 * coeff)
                swift_channel.add_time_operators(j_vec)
                swift_channel.add_swift_operator(j, b)
                measurement.assign(swift_channel)
                swift_channel.shuffle(seed)
                channels.append(swift_channel)

            for s in [0, 1]:
                sign = (-1) ** s
                swift_channel = SwiftChannel(sign * coeff)
                swift_channel.add_time_operators(j_vec)

            for s in [0, 1]:

        return channels

    def _build_compiler(self, operator_pool: OperatorPool, lam):
        tau = lam / self.N
        return Compiler(operator_pool=operator_pool,
                        observables=self.obs.paulis,
                        initializer=self.initializer, nshot=self.shot, tau=tau)
