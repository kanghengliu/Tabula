import argparse
import sys
import os
import pygame

# add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tabula.games import boat, grid_world, geosearch  # Import environments
from tabula.algorithms.dynamic_programming import DynamicProgramming  # Import DynamicProgramming
from tabula.utils import Utils  # Import the Utils class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamic Programming Solver for Boat and GridWorld Environments')
    parser.add_argument('--env', choices=['boat', 'grid_world', 'geosearch'], default='boat',
                        help='Specify the environment to solve: boat or grid_world')
    parser.add_argument('--method', choices=['value', 'policy'], default='value',
                        help='Specify the method to use: value iteration or policy iteration')
    parser.add_argument('--save', action='store_true', help='Save the optimal policy to a file')
    parser.add_argument('--load', action='store_true', help='Load the optimal policy from a file')
    args = parser.parse_args()

    # Choose environment based on the --env flag
    if args.env == 'boat':
        env = boat.BoatEnv()
        episodes = 500
        max_steps = 50
    elif args.env == 'grid_world':
        env = grid_world.GridWorldEnv()
        episodes = 5000
        max_steps = 50
    else:  # geosearch case
        env = geosearch.GeosearchEnv()
        episodes = 1000
        max_steps = 25

    dp_solver = DynamicProgramming(env)

    # Load the policy if specified
    if args.load:
        dp_solver.policy = Utils.load_policy(filename="optimal_policy.pkl")
    else:
        # Run a long simulation to gather p(s', r | s, a)
        dp_solver.simulate(episodes=episodes, max_steps=max_steps)
    
        # Compute the transition model
        transition_model = dp_solver.compute_transition_model()

        # Use value iteration or policy iteration based on the argument
        if args.method == 'value':
            optimal_policy = dp_solver.value_iteration()
            print("Optimal Policy (Value Iteration):")
        else:
            optimal_policy = dp_solver.policy_iteration()
            print("Optimal Policy (Policy Iteration):")

        print(optimal_policy)

        # Save the policy if specified
        if args.save:
            Utils.save_policy(dp_solver.policy, filename="optimal_policy.pkl")

    # Visualize the optimal policy using arrows
    Utils.render_optimal_policy(env, dp_solver.policy)

    # Wait for a key press before continuing
    print("Press any key while in pygame window to continue to visualize the agent running the optimal policy...")
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
    print("Running agent with optimal policy...")
    Utils.run_optimal_policy(env=env, policy=dp_solver.policy)