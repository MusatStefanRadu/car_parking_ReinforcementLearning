import numpy as np
import random
import csv
import os

from environment.environment import create_environment

NUM_BINS = 16


def discretize(value, min_val, max_val, bins=NUM_BINS):
    value = np.clip(value, min_val, max_val)
    idx = int((value - min_val) / (max_val - min_val) * bins)
    return min(idx, bins - 1)



ACTIONS = [
    np.array([ 1.0,  0.0]),
    np.array([-1.0,  0.0]),
    np.array([ 0.0,  1.0]),
    np.array([ 0.0, -1.0]),
    np.array([ 0.0,  0.0])
]

N_ACTIONS = len(ACTIONS)


def choose_action(state, Q, epsilon):
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    return max(range(N_ACTIONS), key=lambda a: Q.get((state, a), 0.0))


def run_sarsa_experiment(
    alpha,
    gamma,
    epsilon_start,
    epsilon_decay,
    seed,
    episodes=500,
    max_steps=150
):


    random.seed(seed)
    np.random.seed(seed)

    env = create_environment(render=False, seed=seed)

    Q = {}  # Q-table
    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(episodes):

        obs, _ = env.reset()
        state = discretize_state(obs)
        action = choose_action(state, Q, epsilon)

        total_reward = 0

        for _ in range(max_steps):

            obs, reward, terminated, truncated, _ = env.step(ACTIONS[action])
            done = terminated or truncated

            next_state = discretize_state(obs)
            next_action = choose_action(next_state, Q, epsilon)

            # formula sarsa
            old_q = Q.get((state, action), 0.0)
            next_q = Q.get((next_state, next_action), 0.0)

            Q[(state, action)] = old_q + alpha * (reward + gamma * next_q - old_q)

            state = next_state
            action = next_action
            total_reward += reward

            if done:
                break

        epsilon = max(0.05, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

    env.close()
    return episode_rewards



def run_all_experiments():

    experiments = [
        {"alpha": 0.05, "gamma": 0.99, "epsilon": 1.0, "decay": 0.995},
        {"alpha": 0.10, "gamma": 0.99, "epsilon": 1.0, "decay": 0.995},
        {"alpha": 0.20, "gamma": 0.99, "epsilon": 1.0, "decay": 0.995},
    ]

    seeds = [0, 42, 123]

    os.makedirs("results", exist_ok=True)

    for exp_id, params in enumerate(experiments):
        for seed in seeds:

            rewards = run_sarsa_experiment(
                alpha=params["alpha"],
                gamma=params["gamma"],
                epsilon_start=params["epsilon"],
                epsilon_decay=params["decay"],
                seed=seed
            )

            filename = f"results/sarsa_exp{exp_id}_seed{seed}.csv"
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(rewards)

            print(f"[OK] Experiment {exp_id}, seed {seed} salvat -> {filename}")

def discretize_state(obs):
    obs_vec = obs["observation"]

    x, y = obs_vec[0], obs_vec[1]
    vx, vy = obs_vec[2], obs_vec[3]

    cos_t, sin_t = obs_vec[4], obs_vec[5]
    theta = np.arctan2(sin_t, cos_t)

    speed = np.sqrt(vx**2 + vy**2)

    return (
        discretize(x, -1, 1),
        discretize(y, -1, 1),
        discretize(theta, -np.pi, np.pi),
        discretize(speed, 0.0, 1.0)
    )


if __name__ == "__main__":
    run_all_experiments()
