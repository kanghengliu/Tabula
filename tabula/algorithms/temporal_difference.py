import numpy as np
import random
import sys
import os
import gymnasium as gym
from gymnasium import spaces
from tabula.utils import Utils  # Import the Utils class


class TemporalDifference:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # Determine the number of states based on observation space
        if isinstance(self.env.observation_space, spaces.Tuple):
            self.num_states = (
                self.env.observation_space[0].n * self.env.observation_space[1].n
            )  # GridWorld case
        else:
            self.num_states = self.env.observation_space.n  # Discrete case for BoatEnv

        # Initialize Q-table for state-action values
        self.q_table = np.zeros((self.num_states, self.env.action_space.n))

    def learn(self, episodes=1000, max_steps=100):
        """Train the agent using SARSA (on-policy TD control)."""
        for episode in range(episodes):
            state, info = self.env.reset()
            state_idx = Utils._state_to_index(self.env, state)
            action = Utils.epsilon_greedy(
                self.env, self.q_table, state_idx, epsilon=self.epsilon
            )

            terminated = False
            steps = 0

            while not terminated:
                # Take action, observe reward and next state
                next_state, reward, terminated, _, _ = self.env.step(action)
                next_state_idx = Utils._state_to_index(self.env, next_state)

                # Choose next action using epsilon-greedy policy
                next_action = Utils.epsilon_greedy(
                    self.env, self.q_table, next_state_idx, epsilon=self.epsilon
                )

                # Update Q-value based on SARSA update rule
                self.q_table[state_idx, action] += self.alpha * (
                    reward
                    + self.gamma * self.q_table[next_state_idx, next_action]
                    - self.q_table[state_idx, action]
                )

                # Move to the next state and action
                state_idx, action = next_state_idx, next_action
                steps += 1

                # Render the environment if desired
                self.env.render()

                # Terminate after max steps
                if steps >= max_steps:
                    terminated = True

            # Log progress every few episodes
            if episode % (episodes // 10) == 0:
                print(f"Episode {episode + 1}/{episodes} completed")

    def derive_policy(self):
        """Derive the optimal policy from the learned Q-table."""
        policy = np.zeros((self.num_states, self.env.action_space.n))
        for s in range(self.num_states):
            best_action = np.argmax(self.q_table[s])
            policy[s] = np.eye(self.env.action_space.n)[best_action]
        return policy
