import gymnasium as gym
import numpy as np


class CustomParkingRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.prev_distance = None
        self.steps = 0
        self.max_steps = 150

        self.distance_threshold = 0.05
        self.orientation_threshold = 0.1
        self.speed_threshold = 0.05

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps = 0
        self.prev_distance = self._distance_to_goal(obs)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        distance = self._distance_to_goal(obs)
        distance_reward = self.prev_distance - distance
        self.prev_distance = distance

        reward = 2.0 * distance_reward

        orientation_error = self._orientation_error(obs)
        reward -= 0.1 * orientation_error

        reward -= 0.05

        if info.get("crashed", False):
            reward -= 10.0
            terminated = True


        speed = self._speed(obs)

        if (
            distance < self.distance_threshold
            and orientation_error < self.orientation_threshold
            and speed < self.speed_threshold
        ):
            reward += 30.0
            terminated = True


        if self.steps >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info


    def _distance_to_goal(self, obs):
        obs_vec = obs["observation"]
        x, y = obs_vec[0], obs_vec[1]
        return np.sqrt(x**2 + y**2)

    def _orientation_error(self, obs):
        obs_vec = obs["observation"]
        cos_t, sin_t = obs_vec[4], obs_vec[5]
        theta = np.arctan2(sin_t, cos_t)
        return abs(theta)

    def _speed(self, obs):
        obs_vec = obs["observation"]
        vx, vy = obs_vec[2], obs_vec[3]
        return np.sqrt(vx**2 + vy**2)
