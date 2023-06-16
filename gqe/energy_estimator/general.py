from qswift.compiler import Compiler, OperatorPool, SwiftChannel
from qswift.measurement import NaiveGenerator
from qwrapper.hamiltonian import Hamiltonian
from qswift.exact import ExactComputation
from qwrapper.hamiltonian import to_matrix_hamiltonian


class GeneralEstimator:
    def __init__(self, pool: OperatorPool, hamiltonian: Hamiltonian, initializer, tau, nshot=0):
        self._pool = pool
        self.hamiltonian = hamiltonian
        self.initializer = initializer
        if nshot == 0:
            self._measurement_gen = NaiveGenerator([1])
            observables = [hamiltonian]
        else:
            self._measurement_gen = NaiveGenerator(hamiltonian.hs)
            observables = hamiltonian.paulis
        self._compiler = Compiler(operator_pool=pool,
                                  observables=observables,
                                  initializer=initializer,
                                  tau=tau, nshot=nshot)

    def exact(self, time_evolution):
        """
        :return: Tr(H e^{i\sum_{j=1}h_j O_j} rho e^{-i\sum_{j=1}h_j O_j})}
        """
        computation = ExactComputation(to_matrix_hamiltonian(self.hamiltonian),
                                       to_matrix_hamiltonian(time_evolution), 1, self.initializer)
        return computation.compute()

    def evaluate(self, indices):
        codes = []
        for m in self._measurement_gen.generate(1):
            swift_channel = SwiftChannel(1)
            swift_channel.add_time_operators(indices)
            m.assign(swift_channel)
            codes.extend(self._compiler.to_string(swift_channel))
        res = 0
        for c in codes:
            res += self._compiler.evaluate(c)
        return res
