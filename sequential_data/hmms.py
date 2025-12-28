import pandas as pd
import pymc as pm
import numpy as np
from hmmlearn import hmm

df = pd.read_csv(pm.get_data("deaths_and_temps_england_wales.csv"))
temp = pd.qcut(df["temp"], q=4, labels=False, duplicates='drop')
death = pd.qcut(df["deaths"], q=3, labels=[0, 1, 2], duplicates='drop')

n_temp_states = len(np.unique(temp))
n_death_obs = 3

# HMM1: Supervised
def hmm1(temp_l, death_l):
    trans_matrix = np.zeros((n_temp_states, n_temp_states))
    for i in range(len(temp_l) - 1):
        trans_matrix[temp_l[i], temp_l[i + 1]] += 1
    trans_matrix = trans_matrix / (trans_matrix.sum(axis=1, keepdims=True) + 1e-10)
    
    emit_matrix = np.zeros((n_temp_states, n_death_obs))
    for i in range(len(temp_l)):
        emit_matrix[temp_l[i], death_l[i]] += 1
    emit_matrix = emit_matrix / (emit_matrix.sum(axis=1, keepdims=True) + 1e-10)
    
    start_n = np.bincount(temp_l, minlength=n_temp_states).astype(float)
    start_n = start_n / start_n.sum()
    
    model = hmm.CategoricalHMM(n_components=n_temp_states, random_state=42)
    model.startprob_ = start_n
    model.transmat_ = trans_matrix
    model.emissionprob_ = emit_matrix
    return model

# HMM2: Unsupervised
def hmm2(death_l):
    model = hmm.CategoricalHMM(n_components=n_temp_states, n_iter=100, random_state=42)
    model.fit(death_l.reshape(-1, 1))
    return model

HMM1 = hmm1(temp.values, death.values)
HMM2 = hmm2(death.values)