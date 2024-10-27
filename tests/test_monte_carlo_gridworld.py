import sys
import os
import time
import numpy as np

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tabula.algorithms.monte_carlo import MonteCarloES
from tabula.games.grid_world import GridWorldEnv
import pygame

# Initialize the GridWorld environment
env = GridWorldEnv()

# Create Monte Carlo ES agent with verbose output enabled and lower discount factor
mc_agent = MonteCarloES(env, discount_factor=5, verbose=True)

# Define the number of episodes and steps
num_episodes = 1000
num_steps = 50

# Run Monte Carlo ES algorithm to calculate the optimal policy
Q, policy = mc_agent.monte_carlo_es(num_episodes, num_steps=num_steps)

# Step 1: Print the Optimal Policy in a readable format
directions = ["North", "South", "West", "East"]
print("\nOptimal Policy:")
for state in policy:
    best_action = np.argmax(policy[state])
    print(f"State {state}: Best Action = {directions[best_action]}")

# Step 2: Run the Game with the Optimal Policy
print("\nRunning the game with the optimal policy...")

# Reset the environment to initialize it
state, info = env.reset()

# Run the environment for a certain number of steps using the optimal policy
num_steps = 1000  # Number of simulation steps
for step in range(num_steps):
    # Check for quitting the Pygame window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()

    # Choose action based on the optimal policy
    action = np.argmax(policy[state])  # Greedy action from the optimal policy

    # Take a step in the environment
    new_state, reward, done, truncated, info = env.step(action)

    # Print what happened during this step
    print(
        f"Count {step + 1}: Eposodic Step: {info}, State: {state}, Action: {directions[action]}, Reward: {reward}, New State: {new_state}"
    )

    # Render the current state (with Pygame)
    env.render()

    # Update state
    state = new_state

    # If the episode ends, reset the environment
    if done:
        state, _ = env.reset()
    elif info >= 100:
        state, _ = env.reset()

    # Frame rate limiting for smoother visuals
    time.sleep(0.1)  # Adjust the delay to control speed of the game

# Close the environment rendering window
env.close()
