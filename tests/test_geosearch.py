import time
import pygame
import sys
import os

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tabula.games import geosearch  # Import your custom environment

# Initialize the environment
env = geosearch.GeosearchEnv()

# Parameters for the test
num_episodes = 5  # Number of episodes to run
max_steps_per_episode = 100  # Maximum steps per episode

try:
    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}")
        state, info = env.reset()  # Reset the environment
        total_reward = 0

        for step in range(max_steps_per_episode):
            # Render the environment
            env.render()

            # Randomly choose an action (0: Up, 1: Down, 2: Left, 3: Right)
            action = env.action_space.sample()

            # Take the action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Accumulate the reward
            total_reward += reward

            # # Print the step information
            # print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}, Total Reward={total_reward:.2f}")

            # If environment is continuous, we won't check for termination, but you can still limit by max_steps
            if terminated or truncated:
                break

            # Add a delay to better visualize the environment rendering
            time.sleep(0.1)

        print(f"Episode {episode + 1} finished with Total Reward: {total_reward:.2f}")

except KeyboardInterrupt:
    print("Simulation interrupted.")

finally:
    # Close the environment window
    env.close()
    pygame.quit()