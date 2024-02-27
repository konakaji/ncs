from experiment.base import GPTQETaskBase
from abc import abstractmethod, ABC
from experiment.finetune import InitializerDelegate

class Task(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def run(self, context: map):
        pass

class Pipeline:
    def __init__(self, tasks) -> None:
        self.tasks = tasks
    
    def run(self, cfg):
        context = {"cfg": cfg}
        for task in self.tasks:
            task.run(context)

class TrainAllTask(Task):
    def __init__(self, gptqe: GPTQETaskBase) -> None:
        super().__init__()
        self.gptqe = gptqe

    def run(self, context: map):
        context["min_indices_dict"] = self.gptqe.train(context["cfg"])


class TrainSingleTask(Task):
    def __init__(self, gptqe: GPTQETaskBase) -> None:
        super().__init__()
        self.gptqe = gptqe

    def run(self, context: map):
        min_indices, energy = self.gptqe.train_single(context["cfg"])
        context["min_indices"] = min_indices
        context["energy"] = energy


# class FineTuneTask(Task):
#     def __init__(self, hamiltonian) -> None:
#         super().__init__()
#         self.hamiltonian = hamiltonian

#     def run(self, context: map):
#         cfg = context["cfg"]
#         pqc = TimeEvolutionPQC(self.hamiltonian.nqubit)
#         vqe = VQE(Energy(self.hamiltonian, self.hamiltonian.nqubit, pqc),
#                        InitializerDelegate(initializer, self.hamiltonian.nqubit, cfg.tool), optimizer)


