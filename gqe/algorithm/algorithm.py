from torch.utils.data import DataLoader
from gqe.mingpt.data import EnergyDataset
from gqe.mingpt.utils import set_seed
import os, torch
from gqe.mingpt.model import GPT
from gqe.mingpt.trainer import Trainer
from gqe.mingpt.callback import DefaultCallback, PrintMonitor, PretrainMonitor, FileMonitor


class Algorithm:
    def __init__(self, key, cost, offset):
        self.key = key
        self.cost = cost
        self.offset = offset

    def get_gpt(self, n_gates, total_n_gates):
        model_config = GPT.get_default_config()
        model_config.model_type = 'gpt2'
        model_config.vocab_size = self.cost.vocab_size()
        model_config.n_gates = n_gates  # The number of gates for each circuit
        model_config.block_size = total_n_gates
        model_config.temperature = 5  # Each gate is generated with probability exp(-temperature * logit)
        model_config.embd_pdrop = 0
        model_config.resid_pdrop = 0
        model_config.attn_pdrop = 0
        model_config.std = 0.02
        model_config.energy_offset = self.offset
        model_config.embd_pdrop = 0
        model_config.resid_pdrop = 0
        model_config.attn_pdrop = 0
        return GPT(model_config, self.cost)

    from gqe.mingpt.trainer import Trainer

    def get_trainer(self, model, max_iters, n_samples=50):
        train_config = Trainer.get_default_config()
        train_config.learning_rate = 5e-7  # the model we're using is so small that we can go a bit faster
        train_config.max_iters = max_iters
        train_config.num_workers = 10
        train_config.n_samples = n_samples
        trainer = Trainer(train_config, model)
        return trainer

    def get_n_gates(self, data_file):
        items = data_file.split('_')
        return int(items[2])

    def to_map(self, data_files):
        m = {}
        for data_file in data_files:
            n_gate = self.get_n_gates(data_file)
            if n_gate not in m:
                m[n_gate] = []
            m[n_gate].append(data_file)
        return m

    def run(self):
        data_files = []
        n_gates = 4
        total_ngates = 40
        model = self.get_gpt(n_gates, total_ngates)
        first = True
        while n_gates <= total_ngates:
            if first:
                n_iteration = 10
                first = False
            else:
                n_iteration = 3
            current_file = self.pretrain(model, data_files, n_gates, total_ngates, max_iters=n_iteration,
                                         batch_size=200)
            temperatures = [10, 30]
            if n_gates >= 20:
                temperatures = [10, 30, 50]
            model = self.train(data_files, n_gates, total_ngates, current_file, temperatures=temperatures,
                               del_temperature=0.01)
            n_gates = n_gates + 4

    def pretrain(self, model, data_files, n_gates, total_ngates, batch_size=50, max_iters=10):
        current_file = f'../saved_models/gptqe_pretrain_{self.key}_{n_gates}_of_{total_ngates}'
        if os.path.exists(current_file):
            return current_file
        if len(data_files) == 0:
            return
        for _, d_files in self.to_map(data_files).items():
            loader = DataLoader(EnergyDataset(d_files), batch_size=batch_size, shuffle=True)
            callback_generator = DefaultCallback(model, monitors=[PretrainMonitor()], del_temperature=0)
            trainer = self.get_trainer(model, max_iters=max_iters, n_samples=100)
            trainer.set_callback('on_batch_end', callback_generator.generate())
            trainer.pretrain(loader)
        torch.save(model.state_dict(), current_file)
        return current_file

    def train(self, data_files, n_gates, total_ngates, current_file=None, max_iters=200,
              seeds=[31, 37, 43, 47, 53], temperatures=[10, 30], del_temperature=0.01):
        for seed in seeds:
            model = self.get_gpt(n_gates, total_ngates)
            if current_file is not None:
                model.load_state_dict(torch.load(current_file))
            for temperature in temperatures:
                set_seed(seed)
                file_name = f"../output/trajectory_{self.key}_{n_gates}_{temperature}_{seed}.json"
                model_name = f'../saved_models/gptqe_train_{self.key}_{n_gates}_of_{total_ngates}_{temperature}_{seed}'
                if os.path.exists(file_name):
                    data_files.append(file_name)
                    model.load_state_dict(torch.load(model_name))
                    print("loaded:" + model_name)
                    continue
                file_monitor = FileMonitor()
                model.temperature = temperature
                trainer = self.get_trainer(model, max_iters=max_iters, n_samples=50)
                trainer.set_callback('on_batch_end', DefaultCallback(model, monitors=[PrintMonitor(), file_monitor],
                                                                     del_temperature=del_temperature).generate())
                trainer.run()
                torch.save(model.state_dict(), model_name)
                file_monitor.save(file_name)
                data_files.append(file_name)
        # Use model with final seed and temperature is used as the reference
        return model
