

import torch
import os
import sys
import time
import logging
import json
import timeit
import cudaq
import numpy as np
import random
from gqe.naive_model.energy_estimator.iid import IIDEstimator
from gqe.naive_model.simple.model import SimpleModel, Ansatz
from gqe.common.initializer import HFStateInitializer
from qwrapper.optimizer import AdamOptimizer, UnitLRScheduler, PrintMonitor, FileMonitor
from qwrapper.hamiltonian import compute_ground_state
from qswift.compiler import Compiler
from gqe.operator_pool.uccsd import UCCSD, generate_molecule
from experiment.molecule import DiatomicMolecularHamiltonian

N = 8000
n_sample = 1000
iter = 1000
lam = 30
nqubit = 4
seed = 30
# choices of the distance between two atoms
distances = [0.5, 0.6, 0.7, 0.7414, 0.8, 0.9, 1.0, 1.5, 2.0]

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


MODEL_FILEBASE = '../saved_models/model_h2_sto3g_{}_{}.json'
ENERGY_FILEBASE = 'energy_h2_sto3g_{}_{}.txt'
OTHER_FILEBASE = '../output/other_h2_sto3g_{}_{}.json'


class MyQSwiftExecutor:
    def __init__(self):
        self.logger = logging.getLogger("qswift.executor.QSwiftExecutor")

    def execute(self, compiler: Compiler, swift_channels):
        strings = []
        start = time.time()
        for swift_channel in swift_channels:
            string = compiler.to_string(swift_channel)
            strings.extend(string)
        middle = time.time()
        self.logger.debug(
            f"to_string ({len(swift_channels)}): {middle - start}")
        values = []
        start = timeit.default_timer()

        numQpus = cudaq.get_target().num_qpus()
        qpuCounter = 0
        for j, string in enumerate(strings):
            value = compiler.evaluate(
                string, qpu_id=qpuCounter % numQpus, parallelObserve=numQpus > 1)
            qpuCounter += 1
            values.append(value)
            if j % 1000 == 0:
                logging.info(f"{j}")
        self.logger.debug(
            f"evaluate ({len(swift_channels)}): {time.time() - middle}")

        if numQpus > 1:
            values = [v.get() for v in values]
        
        end = timeit.default_timer()
        if len(values):
            print('Time = {} sec, G = {}'.format(end-start, np.sum(values)))
        # if len(values): print(values)

        return np.sum(values)


def find_ground_state_energy(distance, seed, ignore_cache=False):

    molecule = generate_molecule("H", "H", distance, "sto-3g")
    # prepare file
    model_output = MODEL_FILEBASE.format(str(distance), seed)
    energy_output = ENERGY_FILEBASE.format(str(distance), seed)
    other_output = OTHER_FILEBASE.format(str(distance), seed)

    if not ignore_cache and os.path.exists(model_output):
        return

    # prepare Hamiltonian
    hamiltonian = DiatomicMolecularHamiltonian(nqubit, molecule)

    ge = compute_ground_state(hamiltonian)
    print("ground state:", ge)

    # prepare operator_pool
    uccsd = UCCSD(4, molecule)
    paulis = uccsd.paulis
    num_operators = len(paulis)
    ansatz = Ansatz([random.gauss(0, 1) for _ in range(num_operators)],
                    paulis, nqubit=nqubit)

    # prepare simple model
    useCUDAQ = True if '--use-cudaq' in sys.argv else False
    estimator = IIDEstimator(hamiltonian,
                             HFStateInitializer(n_electrons=2),
                             N, K=0, tool='cudaq',
                             n_sample=n_sample, n_grad_sample=1,
                             executor=MyQSwiftExecutor()) if useCUDAQ else IIDEstimator(hamiltonian, HFStateInitializer(n_electrons=2),
                                                                                        N, K=0, n_sample=n_sample, n_grad_sample=1,
                                                                                        executor=MyQSwiftExecutor())

    model = SimpleModel(estimator, ansatz, N, lam, n_sample)
    file_monitor = FileMonitor(energy_output)

    monitors = [PrintMonitor(), file_monitor]

    # run
    model.run(AdamOptimizer(maxiter=iter,
              scheduler=UnitLRScheduler(0.01), monitors=monitors))
    for m in monitors:
        m.finalize()

    exit(0)

    with open(model_output, 'w') as f:
        f.write(model.ansatz.toJSON())

    final_energy = file_monitor.values[len(file_monitor.values) - 1][1]

    m = {"distance": distance,
         "exact_energy": ge,
         "computed_energy": final_energy,
         "n_gates": N, "lam": lam, "seed": seed}
    with open(other_output, "w") as f:
        f.write(json.dumps(m))


find_ground_state_energy(.7474, seed, ignore_cache=False)
