from gqe.mingpt.model import GPT
from gqe.mingpt.cost import EnergyCost
from gqe.mingpt.trainer import Trainer
from gqe.mingpt.callback import DefaultCallback
from gqe.util import to_time_evolutions
from gqe.operator_pool.op import ListablePool
from qswift.initializer import CircuitInitializer
from qml.core.pqc import TimeEvolutionPQC
from qml.core.vqe import VQE
from qml.core.function import Energy
from qwrapper.obs import Hamiltonian
from qwrapper.optimizer import AdamOptimizer
import numpy as np
from abc import abstractmethod, ABC
from gqe.common.initializer import PQCInitializer
from gqe.vqa.initializer import InitializerDelegate


class LayerWiseFactory(ABC):
    @abstractmethod
    def generate_train_config(self, layer):
        pass

    @abstractmethod
    def generate_model_config(self, layer):
        pass


class LayerWiseFineTuneTrainer:
    def __init__(self, factory: LayerWiseFactory,
                 hamiltonian: Hamiltonian,
                 initializer: CircuitInitializer,
                 optimizer: AdamOptimizer,
                 pool: ListablePool, taus,
                 num_layers, device, nshot=0, tool="qulacs"):
        self.pqc = TimeEvolutionPQC(hamiltonian.nqubit)
        self.cost = EnergyCost(hamiltonian, PQCInitializer(initializer, self.pqc), pool, taus, nshot, tool, device)
        self.vqe = VQE(Energy(hamiltonian, hamiltonian.nqubit, self.pqc),
                       InitializerDelegate(initializer, hamiltonian.nqubit, tool), optimizer)
        self.factory = factory
        self.num_layers = num_layers
        self.histories = []
        self.monitors = {}
        self.device = device

    def run(self):
        cost = self.cost
        for layer_index in range(self.num_layers):
            print(f"layer: {layer_index + 1} starts running")
            gpt = GPT(self.factory.generate_model_config(layer_index), cost)
            gpt = gpt.to(self.device)
            trainer = Trainer(self.factory.generate_train_config(layer_index), gpt)
            if layer_index in self.monitors:
                trainer.set_callback("on_batch_end", DefaultCallback(gpt, self.monitors[layer_index]).generate())
            trainer.run()
            operators, taus = to_time_evolutions(cost.sequence, gpt.min_indices)
            for operator, tau in zip(operators, taus):
                self.pqc.add_time_evolution(operator, tau)
            self.vqe.exec()
            self.vqe.optimizer = AdamOptimizer(maxiter=self.vqe.optimizer._maxiter)
            self.pqc.thetas = self.pqc.thetas.tolist()
            self.histories.append(gpt)
            self.vqe.optimizer._t = 0

    def set_monitors(self, layer_index, monitors):
        self.monitors[layer_index] = monitors


class LayerWiseTrainer:
    def __init__(self, factory: LayerWiseFactory, cost: EnergyCost, num_layers, device):
        self.factory = factory
        self.cost = cost
        self.num_layers = num_layers
        self.histories = []
        self.monitors = {}
        self.device = device

    def run(self):
        current_prefixes = None
        cost = self.cost
        for layer_index in range(self.num_layers):
            if current_prefixes is not None:
                cost = cost.copy_with(current_prefixes)
            print(f"layer: {layer_index + 1} starts running")
            gpt = GPT(self.factory.generate_model_config(layer_index), cost)
            gpt = gpt.to(self.device)
            trainer = Trainer(self.factory.generate_train_config(layer_index), gpt)
            if layer_index in self.monitors:
                trainer.set_callback("on_batch_end", DefaultCallback(gpt, self.monitors[layer_index]).generate())
            trainer.run()
            prefixes = gpt.min_indices
            if current_prefixes is None:
                current_prefixes = prefixes
            else:
                current_prefixes = np.append(current_prefixes, prefixes)
            cost = cost.copy_with(prefixes=current_prefixes)
            print(f"current_prefixes: {current_prefixes}")
            self.histories.append(gpt)

    def set_monitors(self, layer_index, monitors):
        self.monitors[layer_index] = monitors
