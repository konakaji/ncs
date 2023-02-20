import random

from qwrapper.obs import PauliObservable
from qwrapper.hamiltonian import HeisenbergModel
from qwrapper.optimizer import AdamOptimizer, UnitLRScheduler, PrintMonitor, FileMonitor
from gqe.gqe import GQE
from gqe.ansatz import Ansatz
from gqe.measurement import MeasurementMethod

if __name__ == '__main__':
    N = 200
    n_sample = 100
    ansatz = Ansatz([random.gauss(0, 1) for _ in range(9)],
                    [
                        PauliObservable("XI"), PauliObservable("YI"), PauliObservable("ZI"),
                        PauliObservable("IX"), PauliObservable("IY"), PauliObservable("IZ"),
                        PauliObservable("XX"), PauliObservable("YY"), PauliObservable("ZZ")
                    ])
    hamiltonian = HeisenbergModel(2)
    mes = MeasurementMethod(hamiltonian)
    gqe = GQE(ansatz, mes, N, n_sample, tool="qulacs")
    monitors = [PrintMonitor(), FileMonitor('output/energy.txt')]
    gqe.run(AdamOptimizer(maxiter=300, scheduler=UnitLRScheduler(0.01),
                          monitors=monitors))
    for m in monitors:
        m.finalize()
