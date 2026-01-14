import gymnasium as gym
import highway_env
from environment.custom_parking_env import CustomParkingRewardWrapper


def create_environment(render=False, seed=None):
    env = gym.make(
        "parking-v0",
        render_mode="human" if render else None,
        config={
            "vehicles_count": 6,
            "controlled_vehicles": 1,
        }
    )

    env = CustomParkingRewardWrapper(env)

    if seed is not None:
        env.reset(seed=seed)

    return env
