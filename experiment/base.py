import json
import os.path
import random
import sys

import numpy, math
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
from experiment.const import *


def key(distance):
    v = str(distance).replace(".", "_")
    return v


class GPTQETaskBase(ABC):
    def train(self, cfg):
        fabric = L.Fabric(accelerator="auto", loggers=[self._get_logger(cfg)])
        fabric.seed_everything(cfg.seed)
        fabric.launch()

        min_indices_dict = {}
        distances = cfg.distances
        filename = train_file(cfg)
        m = {}
        if os.path.exists(filename):
            with open(filename) as f:
                for l in f.readlines():
                    items = l.rstrip().split('\t')
                    if len(items) != 2:
                        continue
                    distance, energy = items
                    distance = float(distance)
                    energy = float(energy)
                    m[distance] = energy
        with open(filename, 'w') as f:
            for distance in distances:
                print("distance:", distance)
                if distance in m:
                    min_energy = m[distance]
                    print("already computed, skipped", distance)
                else:
                    indices, min_energy = self.do_train(cfg, distance, fabric)
                    min_indices_dict[str(distance)] = indices
                f.write(f"{distance}\t{min_energy}\n")
        # plt, impath = self.plot_figure(cfg, computed_energies)
        # fabric.log('result', wandb.Image(plt))
        fabric.log('circuit', json.dumps(min_indices_dict))
        return min_indices_dict

    def train_single(self, cfg):
        fabric = L.Fabric(accelerator="auto", loggers=[self._get_logger(cfg)])
        fabric.seed_everything(cfg.seed)
        fabric.launch()

        min_indices, energy = self.do_train(cfg, cfg.distance, fabric)
        fabric.log('circuit', json.dumps(min_indices))
        return min_indices, energy

    def do_train(self, cfg, distance, fabric):
        print(cfg)
        monitor = FileMonitor()
        cost = self.construct_cost(distance, cfg, cfg.print_exact)
        if cfg.dry:
            return None, None
        cfg.vocab_size = cost.vocab_size()
        model = Transformer(cfg, distance, cfg.small)
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
        # state = {"model": model, "optimizer": optimizer, "hparams": model.hparams}
        # fabric.save(cfg.save_dir + f"checkpoint_{distance}.ckpt", state)
        if cfg.save_data:
            monitor.save(trajectory_file(distance))
        indices = min_indices.cpu().numpy().tolist()
        return indices, min_energy

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
        # model.train()
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
        path = pretrain_file(cfg)
        fabric.save(path, state)
        return path

    def construct_cost(self, distance, cfg, print_exact=True):
        molecule = self.get_molecule(distance, cfg)

        hamiltonian = self.get_hamiltonian(molecule, cfg)
        if print_exact:
            k = '.' + to_hash(hamiltonian)
            if os.path.exists(k):
                print("exist!")
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

    def plot_figure(self, cfg, computed_energies, errors=None):
        maker = FigureMaker(self)
        maker.run(cfg, computed_energies, errors)

    def get_hamiltonian(self, molecule, cfg):
        hamiltonian = DiatomicMolecularHamiltonian(cfg.nqubit, molecule, bravyi_kitaev=cfg.is_bravyi)
        return hamiltonian

    @abstractmethod
    def get_molecule(self, distance, cfg):
        pass

    def _get_operator_pool(self, molecule, cfg):
        uccsd = UCCSD(cfg.nqubit, molecule)
        paulis = uccsd.paulis
        identity = ''.join(["I" for _ in range(cfg.nqubit)])
        paulis.append(PauliObservable(identity))
        return DefaultOperatorPool(paulis)

    def _get_logger(self, cfg):
        cfg.run_name = datetime.now() \
            .strftime("{}_{}_run_%m%d_%H_%M".format(cfg.molecule_name, cfg.seed))
        return WandbLogger(
            project=cfg.name,
            name=cfg.run_name,
            log_model=False,
        )


class FigureMaker:
    def __init__(self, task, seeds=[1, 2, 3]):
        self.task = task
        self.exact = Exact(task)
        self.benchmark = Benchmark(task)
        self.seeds = seeds

    def run(self, cfg, computed_energies, errors=None):
        fabric = L.Fabric(accelerator="auto", loggers=[self._get_logger(cfg)])
        fabric.launch()
        distances = cfg.distances

        print("random benchmark")
        if errors is not None:
            self._plot_random(cfg, distances)

        xs2 = []
        ys2 = []

        for i, d in enumerate(distances):
            xs2.append(d)
            ys2.append(computed_energies[i])

        xs, ys, ys3 = self.exact.run(cfg)
        p.plot(xs, ys, label='exact', linewidth=2, color='#333333')
        if errors is None:
            p.plot(distances, computed_energies, label='gpt-qe', marker='o', linewidth=0, color='#008176')
        else:
            p.errorbar(distances, computed_energies, errors, label='gpt-qe', marker='o', linewidth=0, elinewidth=1,
                       color='#008176')
        p.plot(xs, ys3, label='hf', linewidth=2, linestyle="dotted", color='#999999')
        p.xlabel('bond length (angstrom)', fontsize=12)
        p.ylabel('energy value (Hartree)', fontsize=12)
        p.title(f'{cfg.molecule_name} (sto-3g basis, {cfg.nqubit} qubits, {cfg.ngates} tokens)')
        p.legend(fontsize=10, loc='upper right')
        p.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
        impath = image_file(cfg, errors)
        p.savefig(impath)
        fabric.log('result', wandb.Image(p))
        p.clf()

    def _get_logger(self, cfg):
        cfg.run_name = datetime.now() \
            .strftime("{}_{}_run_%m%d_%H_%M".format(cfg.molecule_name, cfg.seed))
        return WandbLogger(
            project=cfg.name,
            name=cfg.run_name,
            log_model=True,
        )

    def _plot_random(self, cfg, xs):
        rs = []
        for seed in self.seeds:
            rs.append(self.benchmark.run(cfg, seed))

        randoms = []
        random_errors = []
        for j, x in enumerate(xs):
            array = [r[j] for r in rs]
            mean = numpy.mean(array)
            error = numpy.std(array) / math.sqrt(len(array))
            randoms.append(mean)
            random_errors.append(error)
        p.errorbar(xs, randoms, random_errors, label='benchmark', marker='x', linewidth=0, elinewidth=1,
                   color='#666666')


class Exact:
    def __init__(self, task, n_bin=50):
        self.task = task
        self.n_bin = n_bin

    def run(self, cfg):
        distances = cfg.distances
        gs_energies = []
        scf_energies = []
        initializer = HFStateInitializer(n_electrons=cfg.n_electrons)
        gs_file = ground_state_file(cfg)
        min_d = distances[0] - 0.1
        max_d = distances[len(distances) - 1] + 0.1
        if not os.path.exists(gs_file):
            with open(gs_file, "w") as f:
                for j in range(self.n_bin):
                    d = min_d + (max_d - min_d) / (self.n_bin - 1) * j
                    molecule = self.task.get_molecule(d, cfg)
                    hamiltonian = self.task.get_hamiltonian(molecule, cfg)
                    ge = compute_ground_state(hamiltonian)
                    scf = hamiltonian.exact_value(initializer.init_circuit(cfg.nqubit, [], "qulacs"))
                    f.write(f"{d}\t{ge}\t{scf}\n")
        ds = []
        with open(gs_file) as f:
            for l in f.readlines():
                d, ge, scf = l.rstrip().split("\t")
                if float(d) < distances[0] - 0.1 or float(d) > distances[len(distances) - 1] + 0.1:
                    continue
                ds.append(float(d))
                gs_energies.append(float(ge))
                scf_energies.append(float(scf))
        return ds, gs_energies, scf_energies


class Benchmark:
    def __init__(self, task):
        self.task = task

    def run(self, cfg, seed):
        random.seed(seed)
        filename = random_file(cfg, seed)
        m = {}
        if os.path.exists(filename):
            with open(filename) as f:
                for l in f.readlines():
                    dist, v = l.rstrip().split("\t")
                    m[float(dist)] = float(v)
        distances = cfg.distances
        computed_energies = []
        for distance in distances:
            if distance in m:
                computed_energies.append(m[distance])
                continue
            cost = self.task.construct_cost(distance, cfg, print_exact=False)
            min = 0
            for _ in range(cfg.max_iters):
                sequences = torch.randint(high=cost.vocab_size(),
                                          size=(cfg.num_samples, cfg.ngates))
                v = torch.min(cost.energy(sequences)).cpu()
                if min > v:
                    min = v
            computed_energies.append(min)
        with open(filename, 'w') as f:
            for dist, energy in zip(distances, computed_energies):
                f.write(f"{dist}\t{energy}\n")
        return computed_energies
