import json
import os.path
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
from gqe.common.util import to_hash
from qswift.compiler import DefaultOperatorPool
from gqe.mingpt.cost import EnergyCost
from gqe.gptqe.monitor import FileMonitor
from gqe.util import get_device
from datetime import datetime
from benchmark.molecule import DiatomicMolecularHamiltonian


def key(distance):
    return str(distance).replace(".", "")


class GPTQEBase(ABC):
    def train(self, cfg):
        fabric = L.Fabric(accelerator="auto", loggers=[self._get_logger(cfg)])
        fabric.seed_everything(cfg.seed)
        fabric.launch()

        computed_energies = []
        min_indices_dict = {}
        distances = cfg.distances
        for distance in distances:
            indices, min_energy = self._do_run(cfg, distance, fabric)
            computed_energies.append(min_energy)
            min_indices_dict[str(distance)] = indices

        plt, impath = self._plot_figure(cfg, computed_energies)
        fabric.log('result', wandb.Image(plt))
        fabric.log('circuit', json.dumps(min_indices_dict))

    def train_single(self, cfg):
        fabric = L.Fabric(accelerator="auto", loggers=[self._get_logger(cfg)])
        fabric.seed_everything(cfg.seed)
        fabric.launch()

        min_indices, energy = self._do_run(cfg, cfg.distance, fabric)
        fabric.log('circuit', json.dumps(min_indices))
        return min_indices, energy

    def pretrain(self, cfg, data_loader):
        fabric = L.Fabric(accelerator="auto", loggers=[self._get_logger(cfg)])
        fabric.seed_everything(cfg.seed)
        fabric.launch()
        model = Transformer(cfg, 'pretrain')
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        model, optimizer = fabric.setup(model, optimizer)
        if 'pretrain' in cfg.check_points:
            print("loaded from the checkpoint")
            cp = fabric.load(cfg.check_points['pretrain'])
            model.load_state_dict(cp["model"])
            optimizer.load_state_dict(cp["optimizer"])
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total trainable params: {pytorch_total_params / 1e6:.2f}M")
        #model.train()
        size = len(data_loader)
        for iter in range(cfg.max_iters):
            current = 0
            for input, energies in data_loader:
                optimizer.zero_grad()
                loss, energies, indices, log_values = model.train_step(indices=input.to(get_device()),
                                                                       energies=energies.to(get_device()))
                if cfg.verbose:
                    print(f"{loss.item()} at {current}/{size} ({iter}/{cfg.max_iters})")
                fabric.log_dict(log_values, step=current + size * iter)
                fabric.backward(loss)
                fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_norm_clip)
                optimizer.step()
                current += 1
        model.set_cost(None)
        state = {"model": model, "optimizer": optimizer, "hparams": model.hparams}
        fabric.save(cfg.save_dir + f"checkpoint_pretrain.ckpt", state)

    def _get_logger(self, cfg):
        cfg.run_name = datetime.now().strftime("run_%m%d_%H_%M")
        cfg.save_dir = f"checkpoints/{cfg.name}/{cfg.run_name}/"
        return WandbLogger(
            project=cfg.name,
            name=cfg.run_name,
            log_model=True,
        )

    def _do_run(self, cfg, distance, fabric):
        print(cfg)
        monitor = FileMonitor()
        cost = self._construct_cost(distance, cfg)
        cfg.vocab_size = cost.vocab_size()
        model = Transformer(cfg, distance)
        model.set_cost(cost)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        model, optimizer = fabric.setup(model, optimizer)
        if key(distance) in cfg.check_points:
            print("loaded from the checkpoint")
            cp = fabric.load(cfg.check_points[key(distance)])
            model.load_state_dict(cp["model"])
            optimizer.load_state_dict(cp["optimizer"])
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total trainable params: {pytorch_total_params / 1e6:.2f}M")
        model.train()
        min_energy = sys.maxsize
        min_indices = None
        for epoch in range(cfg.max_iters):
            optimizer.zero_grad()
            l = None
            for _ in range(cfg.backward_frequency):
                loss, energies, indices, log_values = model.train_step()
                if l is None:
                    l = loss
                else:
                    l = l + loss
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
                fabric.log_dict(log_values, step=epoch)
            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_norm_clip)
            optimizer.step()
            # scheduler.step()
            model.temperature += cfg.del_temperature
        model.set_cost(None)
        state = {"model": model, "optimizer": optimizer, "hparams": model.hparams}
        fabric.save(cfg.save_dir + f"checkpoint_{distance}.ckpt", state)
        #monitor.save(cfg.save_dir + f"trajectory_{distance}.ckpt")
        indices = min_indices.cpu().numpy().tolist()
        return indices, min_energy

    def _construct_cost(self, distance, cfg):
        molecule = self.get_molecule(distance, cfg)

        hamiltonian = self._get_hamiltonian(molecule, cfg)
        k = '.' + to_hash(hamiltonian)
        if os.path.exists(k):
            with open(k) as f:
                ge = float(f.readline())
        else:
            ge = compute_ground_state(hamiltonian)
            with open(k, 'w') as f:
                f.write(str(ge))
        print("ground state:", ge)
        initializer = HFStateInitializer(n_electrons=cfg.n_electrons)
        scf = hamiltonian.exact_value(initializer.init_circuit(cfg.nqubit, [], "qulacs"))
        print("hf state:", scf)
        pool = self._get_operator_pool(molecule, cfg)
        cost = EnergyCost(hamiltonian, initializer, pool, cfg.time_pool)
        return cost

    def _get_operator_pool(self, molecule, cfg):
        uccsd = UCCSD(cfg.nqubit, molecule)
        paulis = uccsd.paulis
        identity = ''.join(["I" for _ in range(cfg.nqubit)])
        paulis.append(PauliObservable(identity))
        return DefaultOperatorPool(paulis)

    def _plot_figure(self, cfg, computed_energies):
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
            hamiltonian = self._get_hamiltonian(molecule, cfg)
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

    def _get_hamiltonian(self, molecule, cfg):
        hamiltonian = DiatomicMolecularHamiltonian(cfg.nqubit, molecule, bravyi_kitaev=cfg.is_bravyi)
        return hamiltonian

    @abstractmethod
    def get_molecule(self, distance, cfg):
        pass
