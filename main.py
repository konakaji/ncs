import random, math

from qwrapper.obs import PauliObservable, Hamiltonian
from qwrapper.optimizer import AdamOptimizer
from gqe.gqe import GQE
from gqe.ansatz import Ansatz
from gqe.measurement import MeasurementMethod

if __name__ == '__main__':
    N = 1000
    n_sample = 100
    ansatz = Ansatz([random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1)],
                    [PauliObservable("X"), PauliObservable("Y"), PauliObservable("Z")])
    hamiltonian = Hamiltonian([1], [PauliObservable("X")], 1)
    mes = MeasurementMethod(hamiltonian)
    gqe = GQE(ansatz, mes, N, n_sample)
    gqe.run(AdamOptimizer(maxiter=100))
