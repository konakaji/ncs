import random, logging

from qwrapper.obs import PauliObservable
from qwrapper.hamiltonian import HeisenbergModel, compute_ground_state
from qwrapper.optimizer import AdamOptimizer, UnitLRScheduler, PrintMonitor, FileMonitor
from qswift.initializer import XBasisInitializer
from gqe.simple_model.model import SimpleModel, Ansatz
from gqe.energy_estimator.iid import IIDEstimator

INPUT_FILENAME = '../saved_models/model_three.json'
OUTPUT_FILENAME = '../saved_models/model_three.json'

if __name__ == '__main__':
    N = 8000
    n_sample = 1000
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
    estimator = IIDEstimator(hamiltonian,
                             XBasisInitializer(),
                             N, K=0, tool='qulacs', n_sample=n_sample, n_grad_sample=1)
    model = SimpleModel(estimator, ansatz, N, lam, n_sample)
    monitors = [PrintMonitor(), FileMonitor('../output/energy.txt')]
    model.run(AdamOptimizer(maxiter=200, scheduler=UnitLRScheduler(0.01), monitors=monitors))
    for m in monitors:
        m.finalize()
    with open(OUTPUT_FILENAME, 'w') as f:
        f.write(model.ansatz.toJSON())
