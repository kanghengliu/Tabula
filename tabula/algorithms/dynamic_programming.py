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

from tabula.utils import Utils  # Import the Utils class

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

    def simulate(self, episodes=1000, max_steps=100):
        """Run a long simulation to gather p(s', r | s, a) and calculate the average reward."""
        total_reward = 0  # Track total reward across all episodes
        
        for episode in range(episodes):
            state, info = self.env.reset()  # Reset environment at the start of each episode
            terminated = False
            episode_reward = 0  # Track reward for this episode
            steps = 0  # Track the number of steps in the episode
            
            while not terminated:
                # Check for window events to keep Pygame active
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        terminated = True
                        self.env.close()
                        return
                
                # Select an action using epsilon-greedy from Utils
                action = Utils.epsilon_greedy(self.env, self.policy, state, self.epsilon)

                # Take the action and observe the result
                next_state, reward, terminated, _, _ = self.env.step(action)

                # Store transition in transition_counts
                state_idx = Utils._state_to_index(self.env, state)
                next_state_idx = Utils._state_to_index(self.env, next_state)
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
                    terminated = True  # End the episode after reaching the step limit

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