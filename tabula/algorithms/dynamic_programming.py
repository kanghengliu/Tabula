import numpy as np
import random
import sys
import os
import pygame
import argparse

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tabula.games import boat

class DynamicProgramming:
    def __init__(self, env, epsilon=0.1, gamma=0.9):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Initialize value function and policy
        self.value_table = np.zeros(self.env.observation_space.n)
        self.policy = np.ones((self.env.observation_space.n, self.env.action_space.n)) / self.env.action_space.n

        # Transition table p(s', r | s, a)
        self.transition_counts = {(s, a): [] for s in range(self.env.observation_space.n) for a in range(self.env.action_space.n)}
    
    def epsilon_greedy(self, state):
        """ Returns an action using epsilon-greedy strategy """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.policy[state])

    def simulate(self, episodes=10000, max_steps=100):
        """ Run a long simulation to gather p(s', r | s, a) """
        for episode in range(episodes):
            state = self.env.reset()  # Reset environment at the start of each episode
            done = False
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
                self.transition_counts[(state, action)].append((next_state, reward))

                # Render the environment after every step
                self.env.render()

                # Update state
                state = next_state

                # Increment step count
                steps += 1
                
                # If max_steps is reached, manually terminate the episode
                if steps >= max_steps:
                    done = True  # End the episode after reaching the step limit

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
        
        Value iteration updates the value function for each state by iteratively 
        finding the maximum expected value over all possible actions. The algorithm 
        continues until the value function converges (i.e., changes between 
        iterations are below a specified tolerance).
        
        Args:
            max_iterations (int): The maximum number of iterations to run.
            tol (float): The convergence threshold. If the difference between value
                        functions in successive iterations is smaller than this 
                        value, the algorithm stops.
        
        Returns:
            numpy.ndarray: The optimal policy after value iteration.
        """
        
        for i in range(max_iterations):
            # Create a copy of the current value table to store updated values
            new_value_table = np.copy(self.value_table)
            
            # Loop through each state in the environment
            for s in range(self.env.observation_space.n):
                action_values = []  # List to store the expected values for each action
                
                # Loop through each possible action in the environment
                for a in range(self.env.action_space.n):
                    q_sa = 0  # Initialize the state-action value
                    
                    # Loop through possible next states and rewards given (s, a)
                    for (next_state, reward), prob in self.transition_model.get((s, a), {}).items():
                        # Bellman equation: expected value of taking action a in state s
                        q_sa += prob * (reward + self.gamma * self.value_table[next_state])
                    
                    # Store the expected value for action a
                    action_values.append(q_sa)
                
                # Update the value of state s to the max of the action values (best action)
                new_value_table[s] = max(action_values)
            
            # Check if the value function has converged
            if np.max(np.abs(new_value_table - self.value_table)) < tol:
                print(f"Value iteration converged at iteration {i}")
                break  # Stop if the value function has converged
            
            # Update the value table with the new values
            self.value_table = new_value_table

        # Once the value iteration is done, derive the optimal policy from the value table
        for s in range(self.env.observation_space.n):
            action_values = []  # List to store the expected values for each action
            
            # Loop through each action in the current state
            for a in range(self.env.action_space.n):
                q_sa = 0  # Initialize the state-action value
                
                # Calculate expected values for (next_state, reward) given (s, a)
                for (next_state, reward), prob in self.transition_model.get((s, a), {}).items():
                    q_sa += prob * (reward + self.gamma * self.value_table[next_state])
                
                # Store the expected value for action a
                action_values.append(q_sa)
            
            # The optimal action is the one with the maximum expected value
            self.policy[s] = np.eye(self.env.action_space.n)[np.argmax(action_values)]

        # Return the optimal policy
        return self.policy


    def policy_iteration(self, max_iterations=1000, tol=1e-3):
        """
        Perform policy iteration to find the optimal policy.
        
        Policy iteration alternates between policy evaluation (calculating the value 
        function for a fixed policy) and policy improvement (updating the policy based 
        on the updated value function). The process continues until the policy is stable.
        
        Args:
            max_iterations (int): The maximum number of iterations to run.
            tol (float): The convergence threshold for the value function.
        
        Returns:
            numpy.ndarray: The optimal policy after policy iteration.
        """
        
        for i in range(max_iterations):
            # === Policy Evaluation Step ===
            # Create a copy of the current value table to update values
            new_value_table = np.copy(self.value_table)
            
            # Loop through each state in the environment
            for s in range(self.env.observation_space.n):
                # Take the action dictated by the current policy
                a = np.argmax(self.policy[s])
                v_s = 0  # state value
                
                # Loop through possible next states and rewards given (s, a)
                for (next_state, reward), prob in self.transition_model.get((s, a), {}).items():
                    # Bellman equation: expected value of taking action a in state s
                    v_s += prob * (reward + self.gamma * self.value_table[next_state])
                
                # Update the value of state s
                new_value_table[s] = v_s
            
            # === Policy Improvement Step ===
            stable_policy = True  # Track whether the policy remains stable
            
            # Loop through each state to improve the policy
            for s in range(self.env.observation_space.n):
                action_values = []  # List to store the expected values for each action
                
                # Loop through each possible action
                for a in range(self.env.action_space.n):
                    q_sa = 0  # Initialize the state-action value
                    
                    # Calculate expected values for (next_state, reward) given (s, a)
                    for (next_state, reward), prob in self.transition_model.get((s, a), {}).items():
                        q_sa += prob * (reward + self.gamma * new_value_table[next_state])
                    
                    # Store the expected value for action a
                    action_values.append(q_sa)
                
                # Determine the best action (the action with the maximum value)
                best_action = np.argmax(action_values)
                
                # If the best action is different from the current policy, update policy
                if np.argmax(self.policy[s]) != best_action:
                    stable_policy = False  # Policy is not stable, needs improvement
                self.policy[s] = np.eye(self.env.action_space.n)[best_action]  # Update policy

            # Update the value table with the new values from policy evaluation
            self.value_table = new_value_table

            # Stop if the policy is stable (no changes in the policy)
            if stable_policy:
                print(f"Policy iteration converged at iteration {i}")
                break  # Policy is stable, iteration can stop

        # Return the optimal policy
        return self.policy
    
    def run_optimal_policy(self, episodes=10, max_steps=100):
        """Simulate episodes where the agent follows the optimal policy and render."""
        for episode in range(episodes):
            state = self.env.reset()  # Reset the environment at the start of each episode
            done = False
            steps = 0

            while not done:
                # Render the environment
                self.env.render()

                # Take the best action according to the optimal policy
                action = np.argmax(self.policy[state])

                # Step in the environment using the optimal action
                next_state, reward, done, _ = self.env.step(action)

                # Update the state
                state = next_state

                # Increment step count
                steps += 1

                # End the episode after reaching the step limit
                if steps >= max_steps:
                    done = True

# === Main Code ===

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Dynamic Programming Solver for Boat Environment')
    parser.add_argument('--method', choices=['value', 'policy'], default='value',
                        help='Specify the method to use: value iteration or policy iteration')
    args = parser.parse_args()

    # Initialize environment
    env = boat.BoatEnv()
    dp_solver = DynamicProgramming(env)

    # Run a long simulation to gather p(s', r | s, a)
    dp_solver.simulate(episodes=5000, max_steps=100)
    
    # Compute the transition model
    transition_model = dp_solver.compute_transition_model()

    print("Transition Model (p(s', r | s, a)):")
    for (s, a), transitions in transition_model.items():
        print(f"State {s}, Action {a}: {transitions}")

    # Use value iteration or policy iteration based on the argument
    if args.method == 'value':
        optimal_policy = dp_solver.value_iteration()
        print("Optimal Policy (Value Iteration):")
    else:
        optimal_policy = dp_solver.policy_iteration()
        print("Optimal Policy (Policy Iteration):")

    print(optimal_policy)

    # Now run the environment again, using the optimal policy and rendering it
    print("Running agent with optimal policy...")
    dp_solver.run_optimal_policy(episodes=1000, max_steps=100)