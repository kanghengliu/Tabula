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
        "--file_path",
        type=str,
        default="./output",
        help="Directory to save GIF and image files",
    )
    parser.add_argument(
        "--save_gif", action="store_true", help="Save gameplay process as GIF"
    )
    parser.add_argument(
        "--save_image",
        action="store_true",
        help="Save optimal policy visualization as an image",
    )
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.file_path, exist_ok=True)

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

    # Save paths for image and GIF
    image_filename = os.path.join(args.file_path, "policy_visualization.png")
    gif_filename = os.path.join(args.file_path, "gameplay.gif")

    # Visualize the optimal policy with an option to save as image
    Utils.render_optimal_policy(
        env, optimal_policy, save_image=args.save_image, image_filename=image_filename
    )

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

    # Run optimal policy and save GIF if required
    print(
        "Running optimal policy with GIF capture..."
        if args.save_gif
        else "Running optimal policy..."
    )
    Utils.run_optimal_policy(
        env, optimal_policy, save_gif=args.save_gif, gif_filename=gif_filename
    )

    print("Done!")
