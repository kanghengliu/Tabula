import numpy as np
import pygame
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces

from tabula.utils import Utils  # Import the Utils class


class DynamicProgramming:
    def __init__(self, env, epsilon=0.1, gamma=0.9):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma

        # Determine the number of states based on observation space
        if isinstance(self.env.observation_space, spaces.Tuple):
            self.num_states = (
                self.env.observation_space[0].n * self.env.observation_space[1].n
            )  # GridWorld case
        else:
            self.num_states = self.env.observation_space.n  # Discrete case for BoatEnv

        # Initialize value function and policy
        self.value_table = np.zeros(self.num_states)
        self.policy = (
            np.ones((self.num_states, self.env.action_space.n))
            / self.env.action_space.n
        )

        # Transition table p(s', r | s, a)
        self.transition_counts = {
            (s, a): []
            for s in range(self.num_states)
            for a in range(self.env.action_space.n)
        }

        # Mean reward and mean value metrics
        self.mean_reward = []
        self.mean_value = []  # To track convergence of the value function

    def simulate(self, episodes, max_steps, verbose=False):
        """Run a long simulation to gather p(s', r | s, a) and calculate the average reward."""
        if verbose:
            print(f"Running simulation for {episodes} episodes...")
        total_reward = 0  # Track total reward across all episodes

        for episode in range(episodes):
            state, info = (
                self.env.reset()
            )  # Reset environment at the start of each episode
            terminated = False
            episode_reward = 0  # Track reward for this episode
            steps = 0  # Track the number of steps in the episode

            while not terminated:
                # Select an action using epsilon-greedy from Utils
                action = Utils.epsilon_greedy(
                    self.env, self.policy, state, epsilon=self.epsilon
                )

                # Take the action and observe the result
                next_state, reward, terminated, _, _ = self.env.step(action)

                # Store transition in transition_counts
                state_idx = Utils._state_to_index(self.env, state)
                next_state_idx = Utils._state_to_index(self.env, next_state)
                self.transition_counts[(state_idx, action)].append(
                    (next_state_idx, reward)
                )

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
            self.mean_reward.append(total_reward)

            if episode % (episodes / 10) == 0 and verbose:
                print(f"Episode {episode + 1}/{episodes}")

        if verbose:
            print("Simulation complete.")
            avg_reward = (
                total_reward / episodes
            )  # Calculate average reward across all episodes
            print(f"\nAverage reward during random simulation: {avg_reward:.2f}")

    def compute_transition_model(self, verbose=False):
        """Computes the transition probabilities p(s', r | s, a) from simulation data"""
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
            self.transition_model[(s, a)] = {
                k: np.round(v / total, 3) for k, v in transition_matrix.items()
            }

        return self.transition_model

    def value_iteration(self, max_iterations=1000, tol=1e-3, verbose=False):
        """Perform value iteration to find the optimal policy."""
        for i in range(max_iterations):
            new_value_table = np.copy(self.value_table)

            for s in range(self.num_states):
                action_values = []

                for a in range(self.env.action_space.n):
                    q_sa = 0
                    for (next_state, reward), prob in self.transition_model.get(
                        (s, a), {}
                    ).items():
                        q_sa += prob * (
                            reward + self.gamma * self.value_table[next_state]
                        )
                    action_values.append(q_sa)

                new_value_table[s] = max(action_values)

            # Track the mean value for convergence plotting
            mean_value = np.mean(new_value_table)
            self.mean_value.append(mean_value)

            # Check for convergence
            if np.max(np.abs(new_value_table - self.value_table)) < tol:
                if verbose:
                    print(f"\nValue iteration converged at iteration {i}")
                break

            self.value_table = new_value_table

        # Derive the optimal policy
        for s in range(self.num_states):
            action_values = []
            for a in range(self.env.action_space.n):
                q_sa = 0
                for (next_state, reward), prob in self.transition_model.get(
                    (s, a), {}
                ).items():
                    q_sa += prob * (reward + self.gamma * self.value_table[next_state])
                action_values.append(q_sa)
            self.policy[s] = np.eye(self.env.action_space.n)[np.argmax(action_values)]

        return self.policy

    def policy_iteration(self, max_iterations=1000, tol=1e-3, verbose=False):
        """Perform policy iteration to find the optimal policy."""
        for i in range(max_iterations):
            new_value_table = np.copy(self.value_table)

            for s in range(self.num_states):
                a = np.argmax(self.policy[s])
                v_s = 0
                for (next_state, reward), prob in self.transition_model.get(
                    (s, a), {}
                ).items():
                    v_s += prob * (reward + self.gamma * self.value_table[next_state])
                new_value_table[s] = v_s

            mean_value = np.mean(new_value_table)
            self.mean_value.append(mean_value)

            stable_policy = True

            for s in range(self.num_states):
                action_values = []
                for a in range(self.env.action_space.n):
                    q_sa = 0
                    for (next_state, reward), prob in self.transition_model.get(
                        (s, a), {}
                    ).items():
                        q_sa += prob * (
                            reward + self.gamma * new_value_table[next_state]
                        )
                    action_values.append(q_sa)
                best_action = np.argmax(action_values)

                if np.argmax(self.policy[s]) != best_action:
                    stable_policy = False
                self.policy[s] = np.eye(self.env.action_space.n)[best_action]

            self.value_table = new_value_table

            if stable_policy:
                if verbose:
                    print(f"\nPolicy iteration converged at iteration {i}")
                break

        return self.policy

    def save_convergence_plot(self, file_path="dp_convergence_plot.png"):
        """Save the convergence plot using Utils."""
        Utils.plot_convergence(self.mean_value, file_path=file_path)

    def train(self, max_steps=100, episodes=1000, verbose=False):
        """Train the agent using dynamic programming."""
        self.simulate(episodes=episodes, max_steps=max_steps, verbose=verbose)
        self.compute_transition_model(verbose=verbose)
        return self.value_iteration()


class MonteCarlo:
    def __init__(self, env, epsilon=0.1, gamma=1.0):
        """
        Initialize the Monte Carlo ES (Exploring Starts) agent.

        Args:
            env: OpenAI Gym environment
            epsilon: Exploration rate for epsilon-greedy policy
            gamma: Discount factor
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize state space size based on environment type
        if hasattr(env, "grid_width") and hasattr(env, "grid_height"):
            self.state_size = env.grid_width * env.grid_height
        else:
            self.state_size = env.observation_space.n

        # Initialize Q-values and policy as numpy arrays
        self.Q = np.zeros((self.state_size, self.env.action_space.n))
        self.policy = (
            np.ones((self.state_size, self.env.action_space.n))
            / self.env.action_space.n
        )
        self.returns = defaultdict(list)
        self.mean_reward = []

    def _get_state_index(self, state):
        """Convert state to index format expected by Utils."""
        return Utils._state_to_index(self.env, state)

    def generate_episode(self, max_steps=100):
        """
        Generates an episode using epsilon-greedy policy.

        Args:
            max_steps: Maximum number of steps per episode

        Returns:
            list of (state, action, reward) tuples
        """
        episode = []
        state = self.env.reset()[0]
        terminated = False
        steps = 0

        while not terminated:
            state_idx = self._get_state_index(state)
            action = Utils.epsilon_greedy(
                self.env, self.Q[state_idx], state, self.epsilon, is_q_values=True
            )

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))

            steps += 1
            if steps >= max_steps:
                terminated = True

            state = next_state

        return episode

    def update_policy(self, state):
        """
        Update policy to be greedy with respect to Q-values.

        Args:
            state: The state to update the policy for
        """
        state_idx = self._get_state_index(state)
        best_action = np.argmax(self.Q[state_idx])
        self.policy[state_idx] = np.zeros(self.env.action_space.n)
        self.policy[state_idx][best_action] = 1.0

    def train(self, max_steps=100, episodes=1000, verbose=False):
        """
        Performs Monte Carlo Exploring Starts training.

        Args:
            max_steps: Maximum steps per episode
            episodes: Number of episodes to train for
            verbose: Whether to print training progress

        Returns:
            Optimal policy as numpy array
        """
        if verbose:
            print(f"Starting Monte Carlo ES training for {episodes} episodes...")

        total_returns = []
        action_counts = np.zeros(self.env.action_space.n)
        q_value_updates = []

        for episode_num in range(1, episodes + 1):
            episode = self.generate_episode(max_steps=max_steps)
            G = 0
            episode_q_updates = []

            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                state_idx = self._get_state_index(state)
                G = self.gamma * G + reward

                if (state_idx, action) not in [
                    (self._get_state_index(x[0]), x[1]) for x in episode[:t]
                ]:
                    old_q = self.Q[state_idx][action]
                    self.returns[(state_idx, action)].append(G)
                    self.Q[state_idx][action] = np.mean(
                        self.returns[(state_idx, action)]
                    )
                    q_update = abs(self.Q[state_idx][action] - old_q)
                    episode_q_updates.append(q_update)
                    self.update_policy(state)

                action_counts[action] += 1

            avg_q_update = np.mean(episode_q_updates) if episode_q_updates else 0
            q_value_updates.append(avg_q_update)
            total_returns.append(G)
            avg_return = np.mean(total_returns[-(episodes // 10) :])
            self.mean_reward.append(avg_return)

            if verbose and episode_num % (episodes // 10) == 0:
                avg_recent_q_update = np.mean(q_value_updates[-(episodes // 10) :])
                print(
                    f"Episode {episode_num}/{episodes} - "
                    f"Average Return: {avg_return:.2f}, "
                    f"Average Q-Value Update: {avg_recent_q_update:.4f}"
                )

        if verbose:
            action_distribution = action_counts / np.sum(action_counts)
            print(
                "\nAction distribution across episodes:",
                {i: f"{p:.3f}" for i, p in enumerate(action_distribution)},
            )
            print(f"Final Average Return: {np.mean(total_returns):.2f}")
            print(f"Final Average Q-Value Update: {np.mean(q_value_updates):.4f}")

        return self.policy


class TemporalDifference:
    def __init__(self, env, epsilon=0.1, gamma=0.9, alpha=0.1):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        # Determine the number of states based on observation space
        if isinstance(self.env.observation_space, spaces.Tuple):
            self.num_states = (
                self.env.observation_space[0].n * self.env.observation_space[1].n
            )
        else:
            self.num_states = self.env.observation_space.n

        # Initialize Q-table for state-action values
        self.q_table = np.zeros((self.num_states, self.env.action_space.n))

        # Track mean Q-value for convergence plot
        self.mean_reward = []

    def learn(self, episodes, max_steps, verbose=False):
        """Train the agent using SARSA (on-policy TD control)."""
        if verbose:
            print(f"Training Temporal Difference algorithm for {episodes} episodes...")

        total_returns = []
        action_counts = np.zeros(self.env.action_space.n)  # Track action distribution

        for episode in range(episodes):
            state, info = self.env.reset()
            state_idx = Utils._state_to_index(self.env, state)

            # Get initial action using epsilon-greedy
            action = Utils.epsilon_greedy(
                self.env,
                self.q_table[state_idx],
                state,
                epsilon=self.epsilon,
                is_q_values=True,
            )

            terminated = False
            steps = 0
            episode_return = 0  # Track total reward in the episode
            q_value_update_total = 0  # Track Q-value updates to assess convergence

            while not terminated and steps < max_steps:
                # Take action, observe reward and next state
                next_state, reward, terminated, _, _ = self.env.step(action)
                next_state_idx = Utils._state_to_index(self.env, next_state)

                # Choose next action using epsilon-greedy
                next_action = Utils.epsilon_greedy(
                    self.env,
                    self.q_table[next_state_idx],
                    next_state,
                    epsilon=self.epsilon,
                    is_q_values=True,
                )

                # Calculate Q-value update
                q_value_update = self.alpha * (
                    reward
                    + self.gamma * self.q_table[next_state_idx, next_action]
                    - self.q_table[state_idx, action]
                )

                # Apply Q-value update
                self.q_table[state_idx, action] += q_value_update
                q_value_update_total += abs(
                    q_value_update
                )  # Sum of absolute Q-value updates

                # Update action count for action distribution tracking
                action_counts[action] += 1

                # Accumulate return for the episode
                episode_return += reward

                # Move to the next state and action
                state = next_state
                state_idx = next_state_idx
                action = next_action

                steps += 1

            # Track return and average Q-value update for convergence insight
            total_returns.append(episode_return)
            avg_q_update = q_value_update_total / steps if steps > 0 else 0
            avg_return = np.mean(total_returns[-(episodes // 10) :])
            self.mean_reward.append(avg_return)

            # Print progress at every 10% of the episodes
            if verbose and episode % (episodes // 10) == 0:
                print(
                    f"Episode {episode + 1}/{episodes} - Average Return: {avg_return:.2f}, "
                    f"Average Q-Value Update: {avg_q_update:.4f}"
                )

        # Display action distribution for final convergence insight
        if verbose:
            action_distribution = action_counts / np.sum(action_counts)
            print(
                "Training complete! Action distribution across episodes:",
                action_distribution,
            )

    def derive_policy(self):
        """Derive the optimal policy from the learned Q-table."""
        policy = np.zeros((self.num_states, self.env.action_space.n))
        for s in range(self.num_states):
            best_action = np.argmax(self.q_table[s])
            policy[s] = np.eye(self.env.action_space.n)[best_action]
        return policy

    def train(self, episodes=1000, max_steps=100, verbose=False):
        """Performs Temporal Difference learning."""
        self.learn(episodes=episodes, max_steps=max_steps, verbose=verbose)
        policy = self.derive_policy()
        return policy

