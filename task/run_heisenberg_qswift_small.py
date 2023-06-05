import os.path
import random

from qwrapper.obs import PauliObservable
from qwrapper.hamiltonian import HeisenbergModel, compute_ground_state
from qwrapper.optimizer import AdamOptimizer, UnitLRScheduler, PrintMonitor, FileMonitor
from qswift.initializer import XBasisInitializer
from gqe.simple_model.model import SimpleModel, Ansatz
from gqe.energy_estimator.qswift import SecondQSwiftEstimator

INPUT_FILENAME = '../saved_models/model_two___.json'
OUTPUT_FILENAME = '../saved_models/model_two___detail.json'

if __name__ == '__main__':
    # logging.getLogger("gqe.energy_estimator.qswift.SecondQSwiftEstimator").setLevel(logging.DEBUG)
    N = 200
    n_sample = 1000
    lam = 7
    nqubit = 2
    hamiltonian = HeisenbergModel(nqubit)
    print(compute_ground_state(hamiltonian))
    if os.path.exists(INPUT_FILENAME):
        with open(INPUT_FILENAME) as f:
            ansatz = Ansatz.fromJSON(f.read())
    else:
        ansatz = Ansatz([random.gauss(0, 0.5) for _ in range(9)],
                        [
                            PauliObservable("XI"), PauliObservable("YI"), PauliObservable("ZI"),
                            PauliObservable("IX"), PauliObservable("IY"), PauliObservable("IZ"),
                            PauliObservable("XX"), PauliObservable("YY"), PauliObservable("ZZ")
                        ], nqubit=nqubit)
    estimator = SecondQSwiftEstimator(hamiltonian,
                                      XBasisInitializer(),
                                      N, K=1, tool='qulacs', n_sample=n_sample, n_grad_sample=1)
    model = SimpleModel(estimator, ansatz, N, lam, n_sample)
    monitors = [PrintMonitor(), FileMonitor('../output/energy.txt')]
    model.run(AdamOptimizer(maxiter=200, scheduler=UnitLRScheduler(0.01), monitors=monitors))
    for m in monitors:
        m.finalize()
    with open(OUTPUT_FILENAME, 'w') as f:
        f.write(model.ansatz.toJSON())
