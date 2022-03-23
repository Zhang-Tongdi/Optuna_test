import logging
import sys
import optuna
import numpy as np
import os
import lk_evaluation_kernel as lk_eval_kernel

def wrap_sample_info(b, a1, a2):
    info = {"b": b, "a1": a1, "a2": a2,}
    return info

def get_objectives(info, base):
    task_folder_name, task_folder_path_mean, task_folder_path_distribution = lk_eval_kernel.make_simulation_project(info, base)
    print("[Done] task project is created")

    lk_eval_kernel.run_simulation_project(task_folder_name, task_folder_path_mean)
    print("[Done] first simulation is finished")
    
    exp_data = np.genfromtxt(os.path.join(base, "data", "exp_Ti0.33Al0.67_900-wavelengths.txt"))
    print("[Done] experimental data is loaded")
    mean_wavelength_diff = lk_eval_kernel.statistic_simulation_project(task_folder_name, task_folder_path_mean, exp_data)
    print("[Done] statistics of mean wavelength is finished")

    if mean_wavelength_diff==200:
        lk_value = 10  
    else:
        lk_eval_kernel.run_simulation_project(task_folder_name, task_folder_path_distribution)
        print("[Done] first simulation is finished")
        lk_value = lk_eval_kernel.lkcalculation_simulation_project(task_folder_name, task_folder_path_distribution, base)
        print("[Done] lk_evaluation is finished")
    return mean_wavelength_diff, lk_value        



def new_study(study_name, base):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, directions=["minimize", "minimize"])

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
    study = optuna.load_study(study_name=study_name, storage=storage_name)

    def objective(trial):
        b = trial.suggest_float("b", 0.1, 5.0)
        a1 = trial.suggest_float("a1", 0.8, 1.78)
        a2 = trial.suggest_float("a2", 0.57, 1.14)

        # computing processes
        info = wrap_sample_info(b, a1, a2)
        print("info is defined")

        return get_objectives(info, base)

    study.optimize(objective, n_trials=367)


if __name__ == "__main__":
    base = os.getcwd()
    new_study("TiAlN_test", base)
    # load_study("TiAlN_test")