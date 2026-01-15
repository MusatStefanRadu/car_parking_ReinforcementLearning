Car Parking Reinforcement Learning

This project explores the use of Reinforcement Learning techniques for solving an autonomous car parking problem in a simulated environment. The objective is to train an agent that can safely and efficiently park a vehicle by interacting with the environment and learning from reward feedback.

The parking task represents a realistic control problem that involves continuous state spaces, sequential decision making, delayed rewards, and safety constraints. Because there is no explicit solution, the problem is well suited for Reinforcement Learning approaches, where the agent must learn a policy through trial and error.

The environment used in this project is based on the parking-v0 scenario from the highway-env library. In order to better control the learning process and highlight differences between algorithms, the original environment was modified through configuration changes and a custom wrapper. The wrapper redefines the reward function, adjusts termination conditions, and introduces a maximum episode length. The reward design encourages progress toward the target parking position, correct orientation, and safe stopping behavior, while penalizing collisions, inefficient motion, and excessive episode duration.

Three Reinforcement Learning algorithms were implemented and compared. SARSA was used as a tabular, on-policy baseline method, relying on a discretized version of the continuous state space. While stable, SARSA learns slowly and is sensitive to the chosen discretization. DQN extends Q-learning by using a neural network to approximate the value function, allowing it to operate directly on continuous observations. It generally achieves better performance than tabular methods, but can exhibit instability depending on hyperparameters and random initialization. PPO represents a modern policy-based approach that directly optimizes the agent’s policy using constrained updates. In this project, PPO proved to be the most stable and best-performing algorithm.

All algorithms were trained for multiple runs using different random seeds in order to evaluate stability and reproducibility. Each training session consisted of 500 episodes, and results were saved and analyzed using reward evolution plots. The comparison highlights clear differences in convergence speed, stability, and final performance across the three approaches.

Overall, the project demonstrates how different Reinforcement Learning paradigms behave on the same control task, emphasizing the advantages of deep and policy-based methods for complex, continuous environments.

The project was implemented in Python using Gymnasium, highway-env, PyTorch, NumPy, and Matplotlib.
