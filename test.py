import argparse
import pygame
import sys

from tabula.environments import *  # Import all environments
from tabula.solvers import *  # Import all solvers
from tabula.utils import Utils  # Import the Utils class

parser = argparse.ArgumentParser(description="RL Solvers for Various Environments")
parser.add_argument(
    "--env",
    choices=["boat", "grid_world", "geosearch"],
    default="boat",
    help="Specify the environment to solve: boat, grid_world, or geosearch",
)
parser.add_argument(
    "--algo",
    choices=["dp", "mc", "td"],
    default="dp",
    help="Specify the algorithm to use: dynamic programming (dp), Monte Carlo (mc), or Temporal Difference (td)",
)
parser.add_argument("--verbose", action="store_true", help="Print verbose output")
args = parser.parse_args()

# set env and parameters
if args.env == "boat":
    env = BoatEnv()
    if args.algo == "dp":
        episodes = 500
        max_steps = 50
    elif args.algo == "mc":
        episodes = 100
        max_steps = 50
    else:  # td
        episodes = 250
        max_steps = 25
elif args.env == "grid_world":
    env = GridWorldEnv()
    if args.algo == "dp":
        episodes = 5000
        max_steps = 50
    elif args.algo == "mc":
        episodes = 20000
        max_steps = 50
    else:  # td
        episodes = 5000
        max_steps = 50
else:  # geosearch case
    env = GeosearchEnv()
    if args.algo == "dp":
        episodes = 1000
        max_steps = 25
    elif args.algo == "mc":
        episodes = 5000
        max_steps = 50
    else:  # td
        episodes = 5000
        max_steps = 37

# initialize solver based on algo
if args.algo == "dp":
    solver = DynamicProgramming(env)
elif args.algo == "mc":
    solver = MonteCarloES(env)
else:
    solver = TemporalDifference(env)

# train agent
policy = solver.train(max_steps=max_steps, episodes=episodes, verbose=args.verbose)

# print policy
print("\nOptimal Policy:")
print(policy)

# render optimal policy
Utils.render_optimal_policy(env, policy)

# Wait for a key press before continuing
print(
    "Press any key while in pygame window to continue to visualize the agent running the optimal policy..."
)
waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            waiting = False
            break
        elif event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

# Run the environment again, using the optimal policy and rendering it
Utils.run_optimal_policy(env=env, policy=policy)