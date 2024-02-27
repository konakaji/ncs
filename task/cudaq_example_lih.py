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
from benchmark.molecule import DiatomicMolecularHamiltonian

cudaq.set_target('nvidia')

N = 40  # n_gates
n_sample = 50
iter = 500
lam = 30
nqubit = 10
seed = 3047
# choices of the distance between two atoms
distances = [1.0, 1.5, 1.57, 2.0, 2.5, 3.0]

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# PRAO: check the location of these files again
MODEL_FILEBASE = '../saved_models/model_lih_sto3g_{}_{}_{}.json'
ENERGY_FILEBASE = 'energy_lih_sto3g_{}_{}_{}.txt'
TRAJECTORY_FILEBASE = '../output/{}_trajectory_lih_sto3g_{}_{}_{}.txt'
OTHER_FILEBASE = '../output/other_lih_sto3g_{}_{}_{}.json'


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
            values = [c * v.get().expectation_z() for (c, v) in values]

        end = timeit.default_timer()
        if len(values):
            print('Time = {} sec, G = {}'.format(end-start, np.sum(values)))
        # if len(values): print(values)

        return np.sum(values)


def find_ground_state_energy(distance, seed, ignore_cache=False):

    transformation = 'jordan-wigner'
    is_bravyi = transformation == 'bravyi-kitaev'
    molecule = generate_molecule("Li", "H", distance, "sto-3g", bravyi_kitaev=is_bravyi)
    # prepare file
    # PRAO: figure out how to do the tranformation for the seed
    model_output = MODEL_FILEBASE.format(str(distance), transformation, seed)
    energy_output = ENERGY_FILEBASE.format(str(distance), transformation, seed)
    other_output = OTHER_FILEBASE.format(str(distance), transformation, seed)
    trajectory_output = TRAJECTORY_FILEBASE.format(model_output, str(distance), transformation, seed)

    if not ignore_cache and os.path.exists(model_output):
        return

    # prepare Hamiltonian
    hamiltonian = DiatomicMolecularHamiltonian(nqubit, molecule, bravyi_kitaev=is_bravyi)

    ge = compute_ground_state(hamiltonian)
    print("ground state:", ge)

    # prepare operator_pool
    uccsd = UCCSD(nqubit, molecule)
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




print (distances)
print (seed)
for d in distances:
    find_ground_state_energy(d, seed, ignore_cache=False)
