import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import os  # For dynamic path management

class GridWorldEnv(gym.Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()

        # Set up grid dimensions
        self.grid_height = 6
        self.grid_width = 6

        # Define action space (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Define observation space (grid positions)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.grid_height), spaces.Discrete(self.grid_width)))

        # Grid reward structure
        self.grid = np.full((self.grid_height, self.grid_width), -1)
        self.grid[1, 4] = -50  # Red cell
        self.grid[5, 0] = -50  # Red cell
        self.grid[5, 5] = 100  # Green cell

        # Terminal states
        self.terminal_states = [(1, 4), (5, 0), (5, 5)]

        # Walls (these positions are impassable)
        self.walls = [(0, 2), (1, 2), (3, 2), (3, 3), (3, 4), (4,2), (5, 2)]

        # Start position
        self.start_pos = (0, 0)

        # Initial agent position
        self.agent_pos = self.start_pos

        # Pygame setup for rendering
        pygame.init()
        self.screen_width = 600
        self.screen_height = 600
        self.cell_size = 100  # Each cell in the grid will be 100x100 pixels
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("GridWorld Environment")
        self.clock = pygame.time.Clock()

        # Dynamic path to assets folder
        current_dir = os.path.dirname(__file__)  # Get the directory of the current file (GridWorldEnv)
        asset_path = os.path.join(current_dir, '../assets/robot.png')  # Build the full path to the robot image

        # Load the robot image
        self.robot_image = pygame.image.load(asset_path)
        self.robot_image = pygame.transform.scale(self.robot_image, (50, 50))  # Scale it to fit

        # Load colors
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'gray': (128, 128, 128),
        }

        # Load font
        self.font = pygame.font.SysFont(None, 24)

    def reset(self):
        """Reset the environment to the initial state."""
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        """Apply the action, return the next state, reward, done, and info."""
        if self.agent_pos in self.terminal_states:
            return self.agent_pos, 0, True, {}

        # Stochastic action: 25% chance of doing the random action, 75% chance of chosen action
        if random.random() > 0.75: # CHANGE TO 0.25 before submission 
            action = self.action_space.sample()

        # Movement logic
        next_pos = self.agent_pos
        i, j = self.agent_pos

        if action == 0 and i > 0:  # Up
            next_pos = (i - 1, j)
        elif action == 1 and i < self.grid_height - 1:  # Down
            next_pos = (i + 1, j)
        elif action == 2 and j > 0:  # Left
            next_pos = (i, j - 1)
        elif action == 3 and j < self.grid_width - 1:  # Right
            next_pos = (i, j + 1)

        # Check if next position is a wall
        if next_pos in self.walls:
            next_pos = self.agent_pos  # Bounce off the wall

        self.agent_pos = next_pos

        # Get the reward for the new state
        reward = self.grid[self.agent_pos]

        # Check if the state is terminal
        done = self.agent_pos in self.terminal_states

        return self.agent_pos, reward, done, {}

    def render(self, mode='human'):
        """Render the grid and agent using Pygame."""
        self.screen.fill(self.colors['white'])

        # Draw the grid and the agent
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                cell_color = self.colors['white']

                if (i, j) in self.walls:
                    cell_color = self.colors['blue']
                elif (i, j) == (1, 4) or (i, j) == (5, 0):  # Red terminal states
                    cell_color = self.colors['red']
                elif (i, j) == (5, 5):  # Green terminal state
                    cell_color = self.colors['green']

                pygame.draw.rect(self.screen, cell_color, pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.colors['black'], pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size), 1)

        # Draw the robot image (agent)
        self.screen.blit(self.robot_image, (self.agent_pos[1] * self.cell_size + self.cell_size // 4, self.agent_pos[0] * self.cell_size + self.cell_size // 4))

        # Update the display
        pygame.display.flip()

        # # Limit the frame rate
        # self.clock.tick(10)

    def close(self):
        """Close the Pygame window."""
        pygame.quit()


# # Testing the environment
# if __name__ == "__main__":
#     env = GridWorldEnv()
#     num_episodes = 100  # Define how many episodes to run
    
#     for episode in range(num_episodes):
#         obs = env.reset()  # Reset environment at the start of each episode
#         total_reward = 0  # Track the total reward per episode
#         # print(f"Episode {episode + 1} starting...")
        
#         done = False
#         while not done:  # Run until the episode is done (i.e., terminal state is reached)
#             env.render()
#             action = env.action_space.sample()  # Take random actions
#             obs, reward, done, info = env.step(action)
#             total_reward += reward
            
#         # print(f"Episode {episode + 1} finished. Total reward: {total_reward}")
    
#     env.close()  # Close the environment when done