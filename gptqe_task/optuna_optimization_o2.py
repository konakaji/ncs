from experiment.configs import get_default_configs
from experiment.optuna import OptunaBase
from experiment.experiment import O2Experiment


def get_o2_configs():
    cfg = get_default_configs()
    # cfg.distances = [1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9]
    cfg.distance = 1.3
    cfg.ngates = 80
    cfg.max_iters = 2000
    cfg.num_samples = 50
    cfg.n_electrons = 8
    cfg.energy_offset = 147
    cfg.nqubit = 12
    cfg.del_temperature = 0.1
    cfg.molecule_name = "O2"
    return cfg



def gptqe_main(
        trials,
        molecule_name,
        n_steps=50,
):
    print("Optuna will be apply to optimize the model")
    print(f" molecule_name {molecule_name}")
    cfg = get_o2_configs()
    cfg.seed = 1
    cfg.ngates = trials.suggest_int("ngates", 40, cfg.ngates, 5)
    cfg.temperature = trials.suggest_float("temperature", 20.0, 50.0, step=5)
    min_indices, energy = O2Experiment().train_single(cfg)

    return energy


if __name__ == "__main__":
    molecule_name = "O2"
    OptunaBase(molecule_name).run(gptqe_main)
