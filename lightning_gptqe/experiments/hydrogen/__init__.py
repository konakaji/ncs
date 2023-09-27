from gqe.operator_pool.uccsd import UCCSD, generate_molecule
from lightning_gptqe.configs.default_configs import get_default_configs
from benchmark.molecule import DiatomicMolecularHamiltonian
from qwrapper.hamiltonian import compute_ground_state
from qwrapper.obs import PauliObservable
from gqe.common.initializer import HFStateInitializer
from qswift.compiler import DefaultOperatorPool
from gqe.mingpt.cost import EnergyCost

import torch
from matplotlib import pyplot as p

# callbacks
# from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

cfg = get_default_configs()

def find_ground_state_energy(distance, cfg):
    molecule = generate_molecule("H", "H", distance, "sto-3g", bravyi_kitaev=cfg.is_bravyi)

    hamiltonian = DiatomicMolecularHamiltonian(cfg.nqubit, molecule, bravyi_kitaev=cfg.is_bravyi)
    ge = compute_ground_state(hamiltonian)
    print("ground state:", ge)


    # prepare operator_pool
    uccsd = UCCSD(cfg.nqubit, molecule)
    paulis = uccsd.paulis
    paulis.append(PauliObservable("IIII"))
    initializer = HFStateInitializer(n_electrons=2)
    scf = hamiltonian.exact_value(initializer.init_circuit(4, [], "qulacs"))
    print("hf state:", scf)

    pool = DefaultOperatorPool(paulis)
    cost = EnergyCost(hamiltonian, initializer, pool,
                        [0.00625, -0.00625, 0.0125, -0.0125, 0.025, -0.025, 0.05, -0.05, 0.1, -0.1])
    return cost

def find_computed_energy(cost, model):
    model.eval()
    model.temperature = 20
    indices, _ = model.generate(torch.zeros(10, 1, dtype=torch.int))
    index = torch.argmin(cost.energy(indices)).item()
    target = indices.numpy()[index]
    return cost.energy(torch.tensor([target])).item()


def plot_figure(cfg, distances, computed_energies):
    min_d = distances[0] - 0.1
    max_d = distances[len(distances) - 1] + 0.1
    n_bin = 100

    xs = []
    ys = []
    ys3 = []
    initializer = HFStateInitializer(n_electrons=2)
    for j in range(n_bin):
        d = min_d + (max_d - min_d) / (n_bin - 1) * j
        molecule = generate_molecule("H", "H", d, "sto-3g", bravyi_kitaev=False)
        hamiltonian = DiatomicMolecularHamiltonian(cfg.nqubit, molecule, bravyi_kitaev=False)
        ge = compute_ground_state(hamiltonian)
        scf = hamiltonian.exact_value(initializer.init_circuit(4, [], "qulacs"))
        xs.append(d)
        ys.append(ge)
        ys3.append(scf)
        
    xs2 = []
    ys2 = []

    for i, d in enumerate(distances):
        xs2.append(d)
        ys2.append(computed_energies[i])

    # p.grid('-')
    p.plot(xs, ys, label='exact', linewidth=1, color='blue')
    p.plot(xs2, ys2, label='computed', marker='x', linewidth=0, color='green')
    p.plot(xs, ys3, label='hf', linewidth=1, color='gray')
    p.xlabel('bond length (angstrom)')
    p.ylabel('energy value (Hartree)')
    p.title('GPT-QE result with H2 Hamiltonian (sto-3g basis)')
    p.legend()
    impath = cfg.save_dir + "result.png"
    p.savefig(impath)
    return p, impath
    

__all__ = ['find_ground_state_energy', 'find_computed_energy', 'plot_figure', 'cfg']