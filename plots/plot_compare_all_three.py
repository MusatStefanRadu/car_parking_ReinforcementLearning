import os
import re
import numpy as np
import matplotlib.pyplot as plt

# root proiect = folderul parinte al folderului plots
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DQN = os.path.join(PROJECT_ROOT, "results")
RESULTS_SARSA = os.path.join(PROJECT_ROOT, "results")
RESULTS_PPO = os.path.join(PROJECT_ROOT, "results_ppo")


def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

def mean_curve(runs):
    min_len = min(len(r) for r in runs)
    runs = [r[:min_len] for r in runs]
    return np.mean(runs, axis=0)

def load_sarsa_all():
    runs = []
    for fn in os.listdir(RESULTS_SARSA):
        if re.match(r"sarsa_exp\d+_seed\d+\.csv$", fn):
            path = os.path.join(RESULTS_SARSA, fn)
            runs.append(np.loadtxt(path, delimiter=","))
    return runs

def load_dqn_all():
    runs = []
    for fn in os.listdir(RESULTS_DQN):
        if re.match(r"dqn_exp\d+_seed\d+\.csv$", fn):
            path = os.path.join(RESULTS_DQN, fn)
            runs.append(np.loadtxt(path, delimiter=","))
    return runs

def load_ppo_all():
    runs = []
    for fn in os.listdir(RESULTS_PPO):
        if re.match(r"ppo_exp\d+_seed\d+\.csv$", fn):
            path = os.path.join(RESULTS_PPO, fn)
            rewards = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(2,))
            runs.append(rewards)
    return runs

sarsa_runs = load_sarsa_all()
dqn_runs = load_dqn_all()
ppo_runs = load_ppo_all()

plt.figure(figsize=(10, 6))

if sarsa_runs:
    sarsa_mean = mean_curve(sarsa_runs)
    plt.plot(moving_average(sarsa_mean), label="SARSA (medie pe exp+seed)")

if dqn_runs:
    dqn_mean = mean_curve(dqn_runs)
    plt.plot(moving_average(dqn_mean), label="DQN (medie pe exp+seed)")

if ppo_runs:
    ppo_mean = mean_curve(ppo_runs)
    plt.plot(moving_average(ppo_mean), label="PPO (medie pe exp+seed)")

plt.xlabel("Episod")
plt.ylabel("Reward total")
plt.title("Comparatie SARSA vs DQN vs PPO (reward mediu)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
