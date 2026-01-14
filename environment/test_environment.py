import time
from environment.environment import create_environment

def main():
    env = create_environment(render=True)

    obs, info = env.reset()

    for step in range(500):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(
                f"[RESET] step={step}, "
                f"reward={reward:.2f}, "
                f"terminated={terminated}, "
                f"truncated={truncated}, "
                f"info={info}"
            )
            obs, info = env.reset()

        time.sleep(0.05)

    env.close()

if __name__ == "__main__":
    main()
