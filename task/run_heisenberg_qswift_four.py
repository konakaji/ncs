import random, logging

from qwrapper.obs import PauliObservable
from qwrapper.hamiltonian import HeisenbergModel, compute_ground_state
from qwrapper.optimizer import AdamOptimizer, UnitLRScheduler, PrintMonitor, FileMonitor
from qswift.initializer import XBasisInitializer
from gqe.simple_model.model2 import SimpleModel, Ansatz
from gqe.energy_estimator.qswift import SecondQSwiftEstimator

INPUT_FILENAME = '../saved_models/model_three.json'
OUTPUT_FILENAME = '../saved_models/model_three.json'

if __name__ == '__main__':
    # logging.getLogger("gqe.energy_estimator.qswift.SecondQSwiftEstimator").setLevel(logging.DEBUG)
    N = 200
    n_sample = 1000
    lam = 25
    nqubit = 4
    hamiltonian = HeisenbergModel(nqubit)
    print(compute_ground_state(hamiltonian))
    ansatz = Ansatz([random.gauss(0, 1) for _ in range(21)],
                    [
                        PauliObservable("XIII"), PauliObservable("YIII"), PauliObservable("ZIII"),
                        PauliObservable("IIXI"), PauliObservable("IIYI"), PauliObservable("IIZI"),
                        PauliObservable("IIXI"), PauliObservable("IIYI"), PauliObservable("IIZI"),
                        PauliObservable("XXII"), PauliObservable("YYII"), PauliObservable("ZZII"),
                        PauliObservable("IXXI"), PauliObservable("IYYI"), PauliObservable("IZZI"),
                        PauliObservable("IIXX"), PauliObservable("IIYY"), PauliObservable("IIZZ"),
                        PauliObservable("XIIX"), PauliObservable("YIIY"), PauliObservable("ZIIZ")
                    ], nqubit=nqubit)
    estimator = SecondQSwiftEstimator(hamiltonian,
                                      XBasisInitializer(),
                                      N, K=0, tool='qulacs', n_sample=n_sample, n_grad_sample=1)
    model = SimpleModel(estimator, ansatz, N, lam, n_sample)
    monitors = [PrintMonitor(), FileMonitor('../output/energy.txt')]
    model.run(AdamOptimizer(maxiter=1000, scheduler=UnitLRScheduler(0.01), monitors=monitors))
    for m in monitors:
        m.finalize()
    with open(OUTPUT_FILENAME, 'w') as f:
        f.write(model.ansatz.toJSON())
