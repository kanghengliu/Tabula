import argparse
import sys
import os
import pygame
import gymnasium as gym

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tabula.games.boat import BoatEnv
from tabula.games.grid_world import GridWorldEnv
from tabula.games.geosearch import GeosearchEnv
from tabula.algorithms.temporal_difference import TemporalDifference
from tabula.utils import Utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Temporal Difference Solver for Boat, GridWorld, and Geosearch Environments"
    )
    parser.add_argument(
        "--env",
        choices=["boat", "grid_world", "geosearch"],
        default="boat",
        help="Specify the environment to solve: boat, grid_world, or geosearch",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--save_gif", action="store_true", help="Save gameplay process as GIF"
    )
    parser.add_argument(
        "--gif_filename",
        type=str,
        default="gameplay.gif",
        help="Filename for the output GIF",
    )
    args = parser.parse_args()

    # Choose the environment based on the --env flag
    if args.env == "boat":
        env = BoatEnv()
    elif args.env == "grid_world":
        env = GridWorldEnv()
    else:  # geosearch case
        env = GeosearchEnv()

    # Initialize the Temporal Difference solver
    td_solver = TemporalDifference(env)

    # Train the agent
    print(f"Training Temporal Difference algorithm on {args.env} environment...")
    td_solver.learn(episodes=args.episodes, max_steps=args.max_steps)

    # Derive and print the optimal policy
    optimal_policy = td_solver.derive_policy()
    print("Optimal Policy:")
    print(optimal_policy)

    # Visualize the optimal policy
    Utils.render_optimal_policy(env, optimal_policy)


    print("Done!")
