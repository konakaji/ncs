import json
import sys
import torch
import lightning as L
import wandb
import matplotlib.pyplot as p
from gqe.gptqe.transformer import Transformer
from pytorch_lightning.loggers import WandbLogger
from abc import ABC, abstractmethod
from gqe.operator_pool.uccsd import UCCSD
from qwrapper.hamiltonian import compute_ground_state
from qwrapper.obs import PauliObservable
from gqe.common.initializer import HFStateInitializer
from qswift.compiler import DefaultOperatorPool
from gqe.mingpt.cost import EnergyCost
from gqe.gptqe.monitor import FileMonitor
from datetime import datetime


class GPTQEBase(ABC):
    def run(self, cfg):
        cfg.run_name = datetime.now().strftime("run_%m%d_%H_%M")
        cfg.save_dir = f"checkpoints/{cfg.name}/{cfg.run_name}/"
        logger = WandbLogger(
            project=cfg.name,
            name=cfg.run_name,
            log_model=True,
        )
        fabric = L.Fabric(accelerator="auto", loggers=[logger])
        fabric.seed_everything(cfg.seed)
        fabric.launch()

        computed_energies = []
        min_indices_dict = {}
        distances = cfg.distances
        for distance in distances:
            monitor = FileMonitor()
            cost = self.construct_cost(distance, cfg)
            cfg.vocab_size = cost.vocab_size()
            model = Transformer(cfg, distance)
            model.set_cost(cost)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
            model, optimizer = fabric.setup(model, optimizer)
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"total trainable params: {pytorch_total_params / 1e6:.2f}M")
            model.train()
            min_energy = sys.maxsize
            min_indices = None
            for epoch in range(cfg.max_iters):
                optimizer.zero_grad()
                loss, energies, indices, log_values = model.train_step()
                monitor.record(epoch, loss, energies, indices)
                for e, indices in zip(energies, indices):
                    energy = e.item()
                    if energy < min_energy:
                        min_energy = e.item()
                        min_indices = indices
                log_values[f"min_energy at {distance}"] = min_energy
                log_values[f"temperature at {distance}"] = model.temperature
                if cfg.verbose:
                    print(f"energies: {energies}")
                    print(f"temperature: {model.temperature}")
                fabric.log_dict(log_values)
                fabric.backward(loss)
                fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_norm_clip)
                optimizer.step()
                # scheduler.step()
                model.temperature += cfg.del_temperature
            model.set_cost(None)
            state = {"model": model, "optimizer": optimizer, "hparams": model.hparams}
            fabric.save(cfg.save_dir + f"checkpoint_{distance}.ckpt", state)
            monitor.save(cfg.save_dir + f"trajectory_{distance}.ckpt")
            min_indices_dict[distance] = min_indices.cpu().numpy().tolist()
            computed_energies.append(min_energy)

        plt, impath = self.plot_figure(cfg, computed_energies)
        fabric.log('result', wandb.Image(plt))
        fabric.log('circuit', json.dumps(min_indices_dict))

    def construct_cost(self, distance, cfg):
        molecule = self.get_molecule(distance, cfg)

        hamiltonian = self.get_hamiltonian(molecule, cfg)
        ge = compute_ground_state(hamiltonian)
        print("ground state:", ge)
        initializer = HFStateInitializer(n_electrons=cfg.n_electrons)
        scf = hamiltonian.exact_value(initializer.init_circuit(cfg.nqubit, [], "qulacs"))
        print("hf state:", scf)
        pool = self.get_operator_pool(molecule, cfg)
        cost = EnergyCost(hamiltonian, initializer, pool, cfg.time_pool)
        return cost

    def get_operator_pool(self, molecule, cfg):
        uccsd = UCCSD(cfg.nqubit, molecule)
        paulis = uccsd.paulis
        identity = ''.join(["I" for _ in range(cfg.nqubit)])
        paulis.append(PauliObservable(identity))
        return DefaultOperatorPool(paulis)

    def plot_figure(self, cfg, computed_energies):
        distances = cfg.distances
        min_d = distances[0] - 0.1
        max_d = distances[len(distances) - 1] + 0.1
        n_bin = 100

        xs = []
        ys = []
        ys3 = []
        initializer = HFStateInitializer(n_electrons=2)
        for j in range(n_bin):
            d = min_d + (max_d - min_d) / (n_bin - 1) * j
            molecule = self.get_molecule(d, cfg)
            hamiltonian = self.get_hamiltonian(molecule, cfg)
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
        p.title(f'GPT-QE result with {cfg.molecule_name} Hamiltonian (sto-3g basis)')
        p.legend()
        impath = cfg.save_dir + "result.png"
        p.savefig(impath)
        return p, impath

    @abstractmethod
    def get_hamiltonian(self, molecule, cfg):
        pass

    @abstractmethod
    def get_molecule(self, distance, cfg):
        pass
