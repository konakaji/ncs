import random

from qwrapper.obs import PauliObservable
from qwrapper.hamiltonian import HeisenbergModel, compute_ground_state
from qwrapper.optimizer import AdamOptimizer, UnitLRScheduler, PrintMonitor, FileMonitor
from qswift.initializer import XBasisInitializer
from gqe.simple_model.model import SimpleModel, Ansatz
from gqe.energy_estimator.qdrift import QDriftEstimator
from gqe.energy_estimator.qswift import SecondQSwiftEstimator

if __name__ == '__main__':
    N = 2000
    n_sample = 100
    lam = 15
    nqubit = 3
    hamiltonian = HeisenbergModel(nqubit)
    print(compute_ground_state(hamiltonian))
    ansatz = Ansatz([random.gauss(0, 0.5) for _ in range(18)],
                    [
                        PauliObservable("XII"), PauliObservable("YII"), PauliObservable("ZII"),
                        PauliObservable("IXI"), PauliObservable("IYI"), PauliObservable("IZI"),
                        PauliObservable("IIX"), PauliObservable("IIY"), PauliObservable("IIZ"),
                        PauliObservable("XXI"), PauliObservable("YYI"), PauliObservable("ZZI"),
                        PauliObservable("IXX"), PauliObservable("IYY"), PauliObservable("IZZ"),
                        PauliObservable("XIX"), PauliObservable("YIY"), PauliObservable("ZIZ"),
                    ], nqubit=nqubit)
    # estimator = SecondQSwiftEstimator(hamiltonian,
    #                                   XBasisInitializer(),
    #                                   N, tool='qulacs', n_sample=n_sample, n_grad_sample=20)
    estimator = QDriftEstimator(hamiltonian, N)
    model = SimpleModel(estimator, ansatz, N, lam, n_sample)
    monitors = [PrintMonitor(), FileMonitor('../output/energy.txt')]
    model.run(AdamOptimizer(maxiter=200, scheduler=UnitLRScheduler(0.01), monitors=monitors))
    for m in monitors:
        m.finalize()
