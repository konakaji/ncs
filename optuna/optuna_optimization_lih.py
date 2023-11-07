import optuna
from functools import partial
from experiment.experiment import LiHExperiment
from experiment.configs import get_default_configs
import sys

def get_lih_configs():
    cfg = get_default_configs()
    cfg.distances = [1.0, 1.5, 1.57, 2.0, 2.5, 3.0]
    cfg.ngates = 40
    cfg.max_iters = 500
    cfg.num_samples = 50
    cfg.n_electrons = 2
    cfg.energy_offset = 7
    cfg.nqubit = 10
    cfg.del_temperature = 0.1
    cfg.molecule_name = "LiH"
    return cfg



    

def qptqe_main(
    trials,
    molecule_name="LiH",
    n_steps=50,
):
    print("Optuna will be apply to optimize the model")
    print(f" molecule_name {molecule_name}")
    cfg = get_lih_configs()
    cfg.distances = [cfg.distances[-2]]
    cfg.max_iters = n_steps
    cfg.ngates = trials.suggest_int("ngates", 10, cfg.ngates,1)
    cfg.temperature = trials.suggest_float("temperature", cfg.temperature,20.0,step=5.0)
    cfg.del_temperature = trials.suggest_float("del_temperature", 0.0, 1.0,step=0.1)
    
    cfg.embd_pdrop =  trials.suggest_float("embd_pdrop", 0.0, 0.99,step=0.01)
    cfg.attn_pdrop =  trials.suggest_float("attn_pdrop", 0.0, 0.99,step=0.01)
    cfg.num_samples = trials.suggest_int("num_samples", 5, 100,5) # we may need to change the bondry for this one
    
    energy = LiHExperiment().run_optuna(cfg)
    
    return energy 

if __name__ == "__main__":
    molecule_name = "LiH"
    storage = f"sqlite:///qptqe_{molecule_name}.db"
    study_name = f"qptqe_{molecule_name}"
    study = optuna.create_study(
        direction="minimize",
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    objective = partial(
            qptqe_main, molecule_name=molecule_name
        )
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))