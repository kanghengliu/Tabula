import numpy as np
import gymnasium as gym
from collections import defaultdict
from tabula.games.grid_world import GridWorldEnv


class MonteCarloES:
    def __init__(self, env, discount_factor=1.0, verbose=False):
        self.env = env
        self.discount_factor = discount_factor
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))  # Q(s, a) table
        self.returns = defaultdict(list)  # Store returns for each state-action pair
        self.policy = defaultdict(
            lambda: np.ones(self.env.action_space.n) / self.env.action_space.n
        )  # Initialize policy
        self.verbose = verbose  # Whether to print verbose messages
        self.action_names = [
            "North",
            "South",
            "West",
            "East",
        ]  # Action names for readability

    def generate_episode(self, max_steps=100):
        """Generates an episode by interacting with the environment using the current policy."""
        episode = []
        state = self.env.reset()[0]  # Reset the environment to get the initial state
        done = False
        steps = 0
        while not done:
            action = np.random.choice(
                np.arange(self.env.action_space.n), p=self.policy[state]
            )
            next_state, reward, done, truncated, info = self.env.step(action)
            episode.append((state, action, reward))

            # if self.verbose:
            #     action_name = self.action_names[action]
            #     print(
            #         f"Generated step: State={state}, Action={action_name}, Reward={reward}, Next State={next_state}"
            #     )

            steps += 1
            if steps > max_steps:
                done = True
            state = next_state  # Move to the next state

        return episode

    def update_policy(self, state):
        """Update the policy to be greedy with respect to the Q-values."""
        best_action = np.argmax(self.Q[state])
        self.policy[state] = np.eye(self.env.action_space.n)[
            best_action
        ]  # One-hot encoding for greedy policy

        if self.verbose:
            action_name = self.action_names[best_action]
            print(f"Policy updated for State {state}: Best Action = {action_name}")

    def monte_carlo_es(self, num_episodes, num_steps=100):
        """Performs Monte Carlo Exploring Starts for the given number of episodes."""
        for episode_num in range(1, num_episodes + 1):
            if self.verbose:
                print(f"\n=== Starting Episode {episode_num} ===")

            episode = self.generate_episode(max_steps=num_steps)
            G = 0  # Initialize return

            # Process the episode in reverse order
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.discount_factor * G + reward

                # Check if state-action pair has appeared earlier in the episode
                if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                    # Append the return to the returns list and update Q
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(
                        self.returns[(state, action)]
                    )  # Average return

                    if self.verbose:
                        action_name = self.action_names[action]
                        print(
                            f"Updated Q-value for State {state}, Action={action_name}: Q[{state}][{action}] = {self.Q[state][action]}"
                        )

                    # Update the policy to be greedy with respect to the updated Q-values
                    self.update_policy(state)

        return self.Q, self.policy
