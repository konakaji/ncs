import random

from qwrapper.obs import PauliObservable, Hamiltonian
from qwrapper.hamiltonian import HeisenbergModel
from qwrapper.optimizer import AdamOptimizer, UnitLRScheduler, PrintMonitor, FileMonitor
from gqe.gqe import GQE
from gqe.simplemodel import SimpleModel, Ansatz
from gqe.estimator import QDriftEstimator
from gqe.measurement import MeasurementMethod

if __name__ == '__main__':
    N = 2000
    n_sample = 100
    lam = 5
    nqubit = 2
    ansatz = Ansatz([random.gauss(0, 0.5) for _ in range(9)],
                    [
                        PauliObservable("XI"), PauliObservable("YI"), PauliObservable("ZI"),
                        PauliObservable("IX"), PauliObservable("IY"), PauliObservable("IZ"),
                        PauliObservable("XX"), PauliObservable("YY"), PauliObservable("ZZ")
                    ], nqubit=nqubit)
    hamiltonian = HeisenbergModel(nqubit)
    mes = MeasurementMethod(hamiltonian)
    estimator = QDriftEstimator(hamiltonian, N, tool='qulacs')
    model = SimpleModel(estimator, ansatz, N, lam, n_sample)
    monitors = [PrintMonitor(), FileMonitor('output/energy.txt')]
    model.run(AdamOptimizer(maxiter=400, scheduler=UnitLRScheduler(0.01), monitors=monitors))
    for m in monitors:
        m.finalize()
