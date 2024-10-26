import numpy as np
import random
import sys
import os
import pygame
import argparse
import gymnasium as gym
from gymnasium import spaces
import pickle

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tabula.games import boat  # Import boat environment
from tabula.games import grid_world  # Import grid_world environment

class DynamicProgramming:
    def __init__(self, env, epsilon=0.1, gamma=0.9):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Determine the number of states based on observation space
        if isinstance(self.env.observation_space, spaces.Tuple):
            self.num_states = self.env.observation_space[0].n * self.env.observation_space[1].n  # GridWorld case
        else:
            self.num_states = self.env.observation_space.n  # Discrete case for BoatEnv
        
        # Initialize value function and policy
        self.value_table = np.zeros(self.num_states)
        self.policy = np.ones((self.num_states, self.env.action_space.n)) / self.env.action_space.n

        # Transition table p(s', r | s, a)
        self.transition_counts = {(s, a): [] for s in range(self.num_states) for a in range(self.env.action_space.n)}

    def _state_to_index(self, state):
        """Converts a (row, col) tuple into a unique index for GridWorld."""
        if isinstance(state, tuple):
            return state[0] * self.env.grid_width + state[1]  # Convert (i, j) to index
        return state  # For environments with Discrete space like BoatEnv

    def epsilon_greedy(self, state):
        """ Returns an action using epsilon-greedy strategy """
        state_idx = self._state_to_index(state)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # exploitation: choose one of the top actions randomly if there is a tie
            max_value = np.max(self.policy[state_idx])  # Find the maximum value in the policy
            top_actions = np.flatnonzero(self.policy[state_idx] == max_value)  # Indices of actions with max value
            return np.random.choice(top_actions)  # Randomly select one of the top actions

    def simulate(self, episodes=1000, max_steps=100):
        """Run a long simulation to gather p(s', r | s, a) and calculate the average reward."""
        total_reward = 0  # Track total reward across all episodes
        
        for episode in range(episodes):
            state = self.env.reset()  # Reset environment at the start of each episode
            done = False
            episode_reward = 0  # Track reward for this episode
            steps = 0  # Track the number of steps in the episode
            
            while not done:
                # Check for window events to keep Pygame active
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        self.env.close()
                        return
                
                # Select an action using epsilon-greedy
                action = self.epsilon_greedy(state)

                # Take the action and observe the result
                next_state, reward, done, _ = self.env.step(action)

                # Store transition in transition_counts
                state_idx = self._state_to_index(state)
                next_state_idx = self._state_to_index(next_state)
                self.transition_counts[(state_idx, action)].append((next_state_idx, reward))

                # Render the environment after every step
                self.env.render()

                # Update state and reward
                state = next_state
                episode_reward += reward  # Accumulate the episode's reward

                # Increment step count
                steps += 1
                
                # If max_steps is reached, manually terminate the episode
                if steps >= max_steps:
                    done = True  # End the episode after reaching the step limit

            total_reward += episode_reward  # Add episode reward to total reward

            if episode % (episodes/10) == 0:
                print(f"Episode {episode + 1}/{episodes}")

        avg_reward = total_reward / episodes  # Calculate average reward across all episodes
        print(f"Average reward during random simulation: {avg_reward:.2f}")

    def compute_transition_model(self):
        """ Computes the transition probabilities p(s', r | s, a) from simulation data """
        self.transition_model = {}
        for (s, a), transitions in self.transition_counts.items():
            transition_matrix = {}
            for next_state, reward in transitions:
                # Count the occurrences of each (next_state, reward) pair
                if (next_state, reward) not in transition_matrix:
                    transition_matrix[(next_state, reward)] = 1
                else:
                    transition_matrix[(next_state, reward)] += 1

            # Normalize to get probabilities
            total = sum(transition_matrix.values())
            self.transition_model[(s, a)] = {k: np.round(v / total, 3) for k, v in transition_matrix.items()}

        # Format and print transition model in a clean way
        print("Transition Model (p(s', r | s, a)):")
        for (s, a), transitions in self.transition_model.items():
            print(f"State {int(s)}, Action {int(a)}:")
            for (next_state, reward), prob in transitions.items():
                print(f"    Next State: {int(next_state)}, Reward: {int(reward)}, Probability: {float(prob)}")

        # Make sure to return the transition model
        return self.transition_model

    def value_iteration(self, max_iterations=1000, tol=1e-3):
        """
        Perform value iteration to find the optimal policy.
        """
        
        for i in range(max_iterations):
            new_value_table = np.copy(self.value_table)  # Create a copy of the value table for updates
            
            # Iterate over each state
            for s in range(self.num_states):
                action_values = []  # Store the expected values for each action
                
                # Iterate over actions
                for a in range(self.env.action_space.n):
                    q_sa = 0  # State-action value
                    
                    # Iterate over possible next states and rewards
                    for (next_state, reward), prob in self.transition_model.get((s, a), {}).items():
                        q_sa += prob * (reward + self.gamma * self.value_table[next_state])
                    action_values.append(q_sa)
                
                # Update the value of state s with the maximum action value
                new_value_table[s] = max(action_values)
            
            # Check for convergence
            if np.max(np.abs(new_value_table - self.value_table)) < tol:
                print(f"Value iteration converged at iteration {i}")
                break  # Exit if the value function has converged
            
            self.value_table = new_value_table  # Update the value table

        # Derive the optimal policy
        for s in range(self.num_states):
            action_values = []
            for a in range(self.env.action_space.n):
                q_sa = 0
                for (next_state, reward), prob in self.transition_model.get((s, a), {}).items():
                    q_sa += prob * (reward + self.gamma * self.value_table[next_state])
                action_values.append(q_sa)
            self.policy[s] = np.eye(self.env.action_space.n)[np.argmax(action_values)]

        return self.policy

    def policy_iteration(self, max_iterations=1000, tol=1e-3):
        """
        Perform policy iteration to find the optimal policy.
        """
        for i in range(max_iterations):
            # === Policy Evaluation Step ===
            new_value_table = np.copy(self.value_table)  # Create a copy of the value table for updates
            
            for s in range(self.num_states):
                a = np.argmax(self.policy[s])  # Action dictated by the current policy
                v_s = 0  # state value
                for (next_state, reward), prob in self.transition_model.get((s, a), {}).items():
                    v_s += prob * (reward + self.gamma * self.value_table[next_state])
                new_value_table[s] = v_s
            
            # === Policy Improvement Step ===
            stable_policy = True
            
            for s in range(self.num_states):
                action_values = []
                for a in range(self.env.action_space.n):
                    q_sa = 0
                    for (next_state, reward), prob in self.transition_model.get((s, a), {}).items():
                        q_sa += prob * (reward + self.gamma * new_value_table[next_state])
                    action_values.append(q_sa)
                best_action = np.argmax(action_values)

                if np.argmax(self.policy[s]) != best_action:
                    stable_policy = False  # The policy has changed
                self.policy[s] = np.eye(self.env.action_space.n)[best_action]
            
            self.value_table = new_value_table  # Update the value table

            if stable_policy:
                print(f"Policy iteration converged at iteration {i}")
                break

        return self.policy
    
    def save_policy(self, filename="optimal_policy.pkl"):
        """Saves the optimal policy to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self.policy, f)
        print(f"Optimal policy saved to {filename}")

    def load_policy(self, filename="optimal_policy.pkl"):
        """Loads the optimal policy from a file."""
        with open(filename, "rb") as f:
            self.policy = pickle.load(f)
        print(f"Optimal policy loaded from {filename}")

    def draw_arrow(self, x, y, direction, size=50, color=(0, 0, 0)):
        """Draws an arrow at the given (x, y) location, pointing in the given direction."""
        half_size = size // 2
        arrow_head_size = size * 0.4  # Adjust size of the arrowhead
        shaft_length = size * 0.3  # Length of the shaft

        # Calculate the start and end points of the shaft
        if direction == 0:  # Left
            start = (x + half_size, y)
            end = (x - shaft_length, y)
            arrow_tip = (x - half_size, y)
            arrow_head = [(arrow_tip[0], arrow_tip[1] - arrow_head_size // 2), 
                          (arrow_tip[0], arrow_tip[1] + arrow_head_size // 2), 
                          (arrow_tip[0] - arrow_head_size, arrow_tip[1])]
        elif direction == 1:  # Right
            start = (x - half_size, y)
            end = (x + shaft_length, y)
            arrow_tip = (x + half_size, y)
            arrow_head = [(arrow_tip[0], arrow_tip[1] - arrow_head_size // 2), 
                          (arrow_tip[0], arrow_tip[1] + arrow_head_size // 2), 
                          (arrow_tip[0] + arrow_head_size, arrow_tip[1])]
        elif direction == 2:  # Up
            start = (x, y + half_size)
            end = (x, y - shaft_length)
            arrow_tip = (x, y - half_size)
            arrow_head = [(arrow_tip[0] - arrow_head_size // 2, arrow_tip[1]), 
                          (arrow_tip[0] + arrow_head_size // 2, arrow_tip[1]), 
                          (arrow_tip[0], arrow_tip[1] - arrow_head_size)]
        elif direction == 3:  # Down
            start = (x, y - half_size)
            end = (x, y + shaft_length)
            arrow_tip = (x, y + half_size)
            arrow_head = [(arrow_tip[0] - arrow_head_size // 2, arrow_tip[1]), 
                          (arrow_tip[0] + arrow_head_size // 2, arrow_tip[1]), 
                          (arrow_tip[0], arrow_tip[1] + arrow_head_size)]

        # Draw the shaft of the arrow (line)
        pygame.draw.line(self.env.screen, color, start, end, 5)

        # Draw the arrowhead (triangle)
        pygame.draw.polygon(self.env.screen, color, arrow_head)

    def render_optimal_policy(self):
        """Renders the optimal policy using arrows for both BoatEnv and GridWorldEnv."""
        
        # Arrow drawing for directions: 0 = Left, 1 = Right, 2 = Up, 3 = Down
        arrow_mapping = {0: 2, 1: 3, 2: 0, 3: 1}  # Left, Right, Up, Down for GridWorld

        # Check if the environment is GridWorldEnv
        if isinstance(self.env, grid_world.GridWorldEnv):
            self.env.screen.fill(self.env.colors['white'])  # Clear the screen

            # Draw the grid
            for i in range(self.env.grid_height):
                for j in range(self.env.grid_width):
                    cell_color = self.env.colors['white']

                    if (i, j) in self.env.walls:
                        cell_color = self.env.colors['blue']
                    elif (i, j) in self.env.terminal_states:  # Red and green terminal states
                        cell_color = self.env.colors['red'] if self.env.grid[i, j] == -50 else self.env.colors['green']

                    # Draw the grid cells
                    pygame.draw.rect(self.env.screen, cell_color, pygame.Rect(j * self.env.cell_size, i * self.env.cell_size, self.env.cell_size, self.env.cell_size))
                    pygame.draw.rect(self.env.screen, self.env.colors['black'], pygame.Rect(j * self.env.cell_size, i * self.env.cell_size, self.env.cell_size, self.env.cell_size), 1)

                    # Draw the optimal action arrow in the cell if it's not a wall or terminal state
                    if (i, j) not in self.env.walls and (i, j) not in self.env.terminal_states:
                        state_idx = i * self.env.grid_width + j  # Convert (i, j) to index
                        if state_idx < len(self.policy):  # Ensure we're within bounds
                            optimal_action = np.argmax(self.policy[state_idx])
                            direction = arrow_mapping[optimal_action]
                            self.draw_arrow(j * self.env.cell_size + self.env.cell_size // 2, i * self.env.cell_size + self.env.cell_size // 2, direction, size=self.env.cell_size * 0.4)

            # Update the display
            pygame.display.flip()

        # For BoatEnv, we visualize using two states and two actions (left or right)
        elif isinstance(self.env, boat.BoatEnv):
            self.env.screen.fill((255, 255, 255))  # White background

            # Draw the boxes (left and right)
            box_size = 150
            pygame.draw.rect(self.env.screen, (173, 216, 230), pygame.Rect(50, 25, box_size, box_size))  # Left box (light blue)
            pygame.draw.rect(self.env.screen, (173, 216, 230), pygame.Rect(200, 25, box_size, box_size))  # Right box (light blue)
            pygame.draw.rect(self.env.screen, (0, 0, 0), pygame.Rect(50, 25, box_size, box_size), 2)  # Left box border
            pygame.draw.rect(self.env.screen, (0, 0, 0), pygame.Rect(200, 25, box_size, box_size), 2)  # Right box border

            # Draw the optimal action arrows in each box
            for state in range(self.env.observation_space.n):
                optimal_action = np.argmax(self.policy[state])  # Get the optimal action for the state
                
                # Draw the arrow in the correct box
                if state == 0:  # Left box
                    center_pos = (50 + box_size // 2, 25 + box_size // 2)
                    self.draw_arrow(center_pos[0], center_pos[1], optimal_action, size=50, color=(0, 0, 0))
                elif state == 1:  # Right box
                    center_pos = (200 + box_size // 2, 25 + box_size // 2)
                    self.draw_arrow(center_pos[0], center_pos[1], optimal_action, size=50, color=(0, 0, 0))

            pygame.display.flip()
    
    def run_optimal_policy(self, episodes=10, max_steps=100, delay_ms=200):
        """Simulate episodes where the agent follows the optimal policy and render, calculate average reward."""
        total_reward = 0  # Track total reward across all episodes
        
        for episode in range(episodes):
            state = self.env.reset()  # Reset the environment at the start of each episode
            done = False
            episode_reward = 0  # Track reward for this episode
            steps = 0

            while not done:
                # Render the environment
                self.env.render()

                # Introduce a delay so you can see the actions slower (delay in milliseconds)
                pygame.time.wait(delay_ms)

                # Take the best action according to the optimal policy
                action = np.argmax(self.policy[self._state_to_index(state)])

                # Step in the environment using the optimal action
                next_state, reward, done, _ = self.env.step(action)

                # Update the state and accumulate reward
                state = next_state
                episode_reward += reward  # Accumulate the episode's reward

                # Increment step count
                steps += 1

                # End the episode after reaching the step limit
                if steps >= max_steps:
                    done = True

            total_reward += episode_reward  # Add episode reward to total reward

        avg_reward = total_reward / episodes  # Calculate average reward across all episodes
        print(f"Average reward following optimal policy: {avg_reward:.2f}")


# ====== MAIN ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamic Programming Solver for Boat and GridWorld Environments')
    parser.add_argument('--env', choices=['boat', 'grid_world'], default='boat',
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
    else:
        env = grid_world.GridWorldEnv()
        episodes = 10000
        max_steps = 100

    dp_solver = DynamicProgramming(env)

    # Load the policy if specified
    if args.load:
        dp_solver.load_policy(filename="optimal_policy.pkl")
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
            dp_solver.save_policy(filename="optimal_policy.pkl")

    # Visualize the optimal policy using arrows
    dp_solver.render_optimal_policy()

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
    dp_solver.run_optimal_policy(episodes=20, max_steps=100)