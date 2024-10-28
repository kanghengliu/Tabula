import sys
import os
import numpy as np
import argparse
import pygame
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tabula.algorithms.monte_carlo import MonteCarloES
from tabula.games import boat, grid_world, geosearch
from tabula.utils import Utils

# Initialize argparse
parser = argparse.ArgumentParser(description="Monte Carlo Solver for Boat, GridWorld, and Geosearch Environments")
parser.add_argument('--env', choices=['boat', 'grid_world', 'geosearch'], default='boat',
                    help="Specify the environment to solve: 'boat', 'grid_world', or 'geosearch'")
args = parser.parse_args()

# Select environment and set parameters
if args.env == 'boat':
    env = boat.BoatEnv()
    num_episodes = 100
    max_steps = 50
elif args.env == 'grid_world':
    env = grid_world.GridWorldEnv()
    num_episodes = 20000
    max_steps = 50
elif args.env == 'geosearch':
    env = geosearch.GeosearchEnv()
    num_episodes = 5000
    max_steps = 50

# Create Monte Carlo ES agent
mc_agent = MonteCarloES(env, gamma=0.9, epsilon=0.1)

# Run Monte Carlo ES algorithm
print(f"Running Monte Carlo ES for {num_episodes} episodes...")
policy = mc_agent.train(episodes=num_episodes, max_steps=max_steps)

# Print the Optimal Policy
print("Optimal Policy:")
if hasattr(env, 'grid_width') and hasattr(env, 'grid_height'):
    for i in range(env.grid_height):
        for j in range(env.grid_width):
            state_idx = i * env.grid_width + j
            best_action = np.argmax(policy[state_idx])
            print(f"State ({i},{j}): Action {best_action}")
else:
    for state in range(env.observation_space.n):
        best_action = np.argmax(policy[state])
        print(f"State {state}: Action {best_action}")
print(policy)

# Initialize pygame
pygame.init()

# Render the Optimal Policy
print("Rendering the Optimal Policy...")
Utils.render_optimal_policy(env, policy)

# Wait for user input
print("Press any key in the pygame window to start simulation with the optimal policy...")
waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
            waiting = False
        elif event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

# Run the game with the Optimal Policy using Utils
print("\nRunning the game with the optimal policy...")
Utils.run_optimal_policy(env, policy, episodes=10, max_steps=max_steps, delay_ms=100)

# Cleanup
env.close()
pygame.quit()