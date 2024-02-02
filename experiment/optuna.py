import optuna
import threading
import functools
from optuna.trial import TrialState
from functools import partial



class OptunaBase:
    def __init__(self, molecule_name):
        self.molecule_name = molecule_name

    def create_or_load(self):
        storage = f"sqlite:///gptqe_{self.molecule_name}.db"
        study_name = f"gptqe_{self.molecule_name}"
        study = optuna.create_study(
            direction="minimize",
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )
        return study

    def run(self, gptqe_main):
        objective = partial(
            gptqe_main, molecule_name=self.molecule_name
        )
        study = self.create_or_load()
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
