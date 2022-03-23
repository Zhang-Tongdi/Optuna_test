import logging
import sys
import optuna
import matplotlib.pyplot as plt 
import numpy as np
import os

logfile = open("sample.txt", "w")

def obj(a, b):
    return (a-10)**2 + (b-10)**2, (a+10)**2 + (b+10)**2

def new_study(study_name, seed = None):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///{}.db".format(study_name)
    sampler = optuna.samplers.NSGAIISampler(seed = seed)
    study = optuna.create_study(study_name=study_name, storage=storage_name, sampler = sampler, directions=["minimize", "minimize"])

    def objective(trial):
        a = trial.suggest_float("a", -1000, 1000)
        b = trial.suggest_float("b", -1000, 1000)

        print("a = {}, b = {}". format(a, b))
        logfile.write("a = {}, b = {}\n". format(a, b))
        
        return obj(a, b)

    study.optimize(objective, n_trials=200)

if __name__=="__main__":
    new_study("benchmark")