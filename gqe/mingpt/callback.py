from abc import abstractmethod
import torch, json


class Monitor:
    @abstractmethod
    def record(self, model, trainer, detail):
        pass


class PrintMonitor(Monitor):
    def record(self, model, trainer, detail):
        print(
            f"iter_dt {trainer.iter_dt:.2f}s; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f} temperature: {model.temperature}")
        print("mean_logits", torch.mean(detail.logits, 1) * model.energy_scaling)
        print("energies:", detail.energies)
        print("mean:", torch.mean(detail.energies))


class FileMonitor(Monitor):
    def __init__(self):
        self.lines = []

    def record(self, model, trainer, detail):
        line = {
            "iter": trainer.iter_num,
            "loss": trainer.loss.item(),
            "indices": detail.indices.numpy().tolist(),
            "energies": detail.energies.numpy().tolist()
        }
        self.lines.append(line)

    def save(self, path):
        with open(path, 'w') as f:
            for l in self.lines:
                f.write(f"{json.dumps(l)}\n")


class DefaultCallback:
    def __init__(self, model, monitors=None, add_temperature_frequency=1, del_temperature=0.05):
        self.model = model
        self.monitors = []
        if monitors is not None:
            self.monitors = monitors
        self.add_temperature_frequency = add_temperature_frequency
        self.del_temperature = del_temperature

    def generate(self):
        def batch_end_callback(trainer, detail):
            if trainer.iter_num % 1 == 0:
                for monitor in self.monitors:
                    monitor.record(self.model, trainer, detail)
            if trainer.iter_num % self.add_temperature_frequency == 0:
                self.model.temperature += self.del_temperature

        return batch_end_callback
