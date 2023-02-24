import random

from qwrapper.obs import PauliObservable, Hamiltonian
from qwrapper.hamiltonian import HeisenbergModel
from qwrapper.optimizer import AdamOptimizer, UnitLRScheduler, PrintMonitor, FileMonitor
from gqe.gqe import GQE
from gqe.ansatz import Ansatz
from gqe.measurement import MeasurementMethod

if __name__ == '__main__':
    N = 500
    n_sample = 100
    ansatz = Ansatz([random.gauss(0, 0.5) for _ in range(9)],
                    [
                        PauliObservable("XI"), PauliObservable("YI"), PauliObservable("ZI"),
                        PauliObservable("IX"), PauliObservable("IY"), PauliObservable("IZ"),
                        PauliObservable("XX"), PauliObservable("YY"), PauliObservable("ZZ")
                    ])
    hamiltonian = HeisenbergModel(2)
    mes = MeasurementMethod(hamiltonian)
    gqe = GQE(ansatz, mes, N, 0.01, n_sample, tool="qulacs")
    monitors = [PrintMonitor(), FileMonitor('output/energy.txt')]
    gqe.run(AdamOptimizer(maxiter=100, scheduler=UnitLRScheduler(0.01),
                          monitors=monitors))
    for m in monitors:
        m.finalize()
