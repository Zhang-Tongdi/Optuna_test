import logging
import sys
import optuna 
import numpy as np
import os
from kernel import make_simulation_project

def wrap_sample_info(b, a1, a2):
    info = {"b": b, "a1": a1, "a2": a2,}
    return info

def get_objectives(info, base):
    mean_wavelength_diff, kl_value = make_simulation_project(info, base)
    return mean_wavelength_diff, kl_value        


def new_study(study_name, base):
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    sampler = optuna.samplers.NSGAIISampler()
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, sampler = sampler, storage=storage_name, directions=["minimize", "minimize"])

    def objective(trial):
        b = trial.suggest_float("b", 0.1, 5.0)
        a1 = trial.suggest_float("a1", 0.8, 1.78)
        a2 = trial.suggest_float("a2", 0.57, 1.14)

        # computing processes
        info = wrap_sample_info(b, a1, a2)
        print("info is defined")

        return get_objectives(info, base)

    study.optimize(objective, n_trials=500)


def load_study(study_name):
    storage_name = "sqlite:///{}.db".format(study_name)
    sampler = optuna.samplers.NSGAIISampler()
    study = optuna.load_study(study_name=study_name, sampler = sampler, storage=storage_name)

    def objective(trial):
        b = trial.suggest_float("b", 0.1, 5.0)
        a1 = trial.suggest_float("a1", 0.8, 1.78)
        a2 = trial.suggest_float("a2", 0.57, 1.14)

        # computing processes
        info = wrap_sample_info(b, a1, a2)
        print("info is defined")

        return get_objectives(info, base)

    study.optimize(objective, n_trials=440)
    

if __name__ == "__main__":
    base = os.getcwd()
    new_study("TiAlN_test", base)
    # load_study("TiAlN_test")