import numpy as np
import gymnasium as gym
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tabula.utils import Utils

class MonteCarloES:
    def __init__(self, env, gamma=1.0, epsilon=0.1, verbose=False):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.verbose = verbose
        
        # Initialize state space size based on environment type
        if hasattr(env, 'grid_width') and hasattr(env, 'grid_height'):
            self.state_size = env.grid_width * env.grid_height
        else:
            self.state_size = env.observation_space.n
        
        # Initialize Q-values and policy as numpy arrays
        self.Q = np.zeros((self.state_size, self.env.action_space.n))
        self.policy = np.ones((self.state_size, self.env.action_space.n)) / self.env.action_space.n
        self.returns = defaultdict(list)

    def _get_state_index(self, state):
        """Convert state to index format expected by Utils."""
        return Utils._state_to_index(self.env, state)

    def generate_episode(self, max_steps=100):
        """Generates an episode using epsilon-greedy policy."""
        episode = []
        state = self.env.reset()[0]
        terminated = False
        steps = 0
        
        while not terminated:
            state_idx = self._get_state_index(state)
            action = Utils.epsilon_greedy(
                self.env, 
                self.Q[state_idx], 
                state, 
                self.epsilon, 
                is_q_values=True
            )
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))
            
            if self.verbose:
                print(f"Step {steps}: State={state}, Action={action}, Reward={reward}")
            
            steps += 1
            if steps >= max_steps:
                terminated = True
                
            state = next_state
            
        return episode

    def update_policy(self, state):
        """Update policy to be greedy with respect to Q-values."""
        state_idx = self._get_state_index(state)
        best_action = np.argmax(self.Q[state_idx])
        self.policy[state_idx] = np.zeros(self.env.action_space.n)
        self.policy[state_idx][best_action] = 1.0
        
        if self.verbose:
            print(f"Policy updated for State {state}: Best Action = {best_action}")

    def monte_carlo_es(self, num_episodes, max_steps=100):
        """Performs Monte Carlo Exploring Starts."""
        for episode_num in range(1, num_episodes + 1):
            if self.verbose:
                print(f"\n=== Starting Episode {episode_num} ===")
                
            episode = self.generate_episode(max_steps=max_steps)
            G = 0
            
            # Process episode in reverse order
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                state_idx = self._get_state_index(state)
                G = self.gamma * G + reward
                
                # Check if state-action pair appeared earlier
                if (state_idx, action) not in [(self._get_state_index(x[0]), x[1]) for x in episode[:t]]:
                    self.returns[(state_idx, action)].append(G)
                    self.Q[state_idx][action] = np.mean(self.returns[(state_idx, action)])
                    
                    if self.verbose:
                        print(f"Updated Q-value for State {state}, Action={action}: Q[{state_idx}][{action}] = {self.Q[state_idx][action]}")
                    
                    # Update policy
                    self.update_policy(state)
        
        return self.Q, self.policy