import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import os

class GeosearchEnv(gym.Env):
    def __init__(self):
        super(GeosearchEnv, self).__init__()

        # Set up grid dimensions
        self.grid_height = 25
        self.grid_width = 25

        # Define action space (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Define observation space (grid positions)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.grid_height), spaces.Discrete(self.grid_width)))

        # Gaussian parameters for water and gold distributions
        self.mu_water = np.array([20, 20])  # Mean for water distribution
        self.sigma_water = np.array([[1, 0.25], [0.25, 1]])  # Covariance matrix for water
        self.mu_gold = np.array([10, 10])  # Mean for gold distribution
        self.sigma_gold = np.array([[1, -0.25], [-0.25, 1]])  # Covariance matrix for gold

        # User-defined weight for reward calculation
        self.A = 0.75

        # Resource matrices for the grid
        self.water_distribution = self._create_distribution(self.mu_water, self.sigma_water)
        self.gold_distribution = self._create_distribution(self.mu_gold, self.sigma_gold)

        # Create the reward matrix based on the weighted sum of water and gold
        self.reward_matrix = self.A * self.water_distribution + (1 - self.A) * self.gold_distribution

        # Initial agent position will be randomized at the start of each episode
        self.agent_pos = None

        # Pygame setup for rendering
        pygame.init()
        self.screen_width = 875  # Larger grid visualization
        self.screen_height = 875
        self.cell_size = 35  # Each cell in the grid will be 30x30 pixels
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Geosearch Environment")
        self.clock = pygame.time.Clock()

        # Dynamic path to assets folder
        current_dir = os.path.dirname(__file__)  # Get the directory of the current file
        asset_path = os.path.join(current_dir, '../assets/robot.png')  # Path to the robot image

        # Load the robot image
        self.robot_image = pygame.image.load(asset_path)
        self.robot_image = pygame.transform.scale(self.robot_image, (20, 20))  # Scale it to fit the cells

        # Load fixed background colors
        self.background_colors = ['#489030', '#5d924d', '#789030', '#a5803d']
        self.background_color_grid = self._generate_background_grid()

        # Load colors
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'water': (15, 94, 156),
            'gold': (255,207, 64)
        }

    def _generate_background_grid(self):
        """Generate a fixed grid of background colors using the specified color palette."""
        color_grid = np.empty((self.grid_height, self.grid_width), dtype=tuple)
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                # Randomly select a color from the predefined palette
                hex_color = random.choice(self.background_colors)
                rgb_color = pygame.Color(hex_color)
                color_grid[i, j] = rgb_color
        return color_grid

    def _create_distribution(self, mu, sigma):
        """Create a 2D Gaussian distribution over the grid."""
        x = np.arange(0, self.grid_width)
        y = np.arange(0, self.grid_height)
        x, y = np.meshgrid(x, y)
        pos = np.dstack((x, y))
        
        # Gaussian formula for 2D
        n = 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
        inv_sigma = np.linalg.inv(sigma)
        diff = pos - mu
        exponent = -0.5 * np.einsum('...k,kl,...l->...', diff, inv_sigma, diff)
        
        return n * np.exp(exponent)

    def reset(self):
        """Reset the environment to the initial state."""
        self.agent_pos = (random.randint(0, self.grid_height - 1), random.randint(0, self.grid_width - 1))

        # make info an empty dictionary
        info = {}

        return self.agent_pos, info

    def step(self, action):
        """Apply the action, return the next state, reward, done, and info."""
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

        self.agent_pos = next_pos

        # Get the reward for the new state
        reward = self.reward_matrix[self.agent_pos]

        # No terminal states, continuing environment
        terminated = False
        truncated = False  # Truncation flag

        return self.agent_pos, reward, terminated, truncated, {}

    def render(self, mode='human'):
        """Render the grid and agent using Pygame."""
        self.screen.fill(self.colors['white'])

        # Define thresholds for water and gold visibility
        water_threshold = 0.01  # Adjust as needed
        gold_threshold = 0.01   # Adjust as needed

        # Maximum value scaling for color adjustment
        max_water_value = np.max(self.water_distribution)
        max_gold_value = np.max(self.gold_distribution)

        # Draw the grid with fixed background colors
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                # Determine cell color based on resource visibility
                water_value = self.water_distribution[i, j]
                gold_value = self.gold_distribution[i, j]

                # Minimum scale factor to avoid complete blackness
                min_scale_factor = 0.4  # Adjust this value as needed (0.3 gives 30% darkness)

                if water_value > water_threshold:
                    # Darken water color based on its value (higher = darker)
                    scale_factor = max(1 - (water_value / max_water_value), min_scale_factor)  # Ensure it's not below the min
                    base_water_color = np.array(self.colors['water'])
                    darkened_water_color = (base_water_color * scale_factor).astype(int)
                    cell_color = tuple(darkened_water_color)
                elif gold_value > gold_threshold:
                    # Darken gold color based on its value (higher = darker)
                    scale_factor = max(1 - (gold_value / max_gold_value), min_scale_factor)  # Ensure it's not below the min
                    base_gold_color = np.array(self.colors['gold'])
                    darkened_gold_color = (base_gold_color * scale_factor).astype(int)
                    cell_color = tuple(darkened_gold_color)
                else:
                    # Use the pre-generated background color for each cell
                    cell_color = self.background_color_grid[i, j]

                # Draw the cell
                pygame.draw.rect(self.screen, cell_color, pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.colors['black'], pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size), 1)

        # Draw the robot image (agent)
        self.screen.blit(self.robot_image, (self.agent_pos[1] * self.cell_size + self.cell_size // 4, self.agent_pos[0] * self.cell_size + self.cell_size // 4))

        # Update the display
        pygame.display.flip()

        # Limit the frame rate
        self.clock.tick(10)

    def close(self):
        """Close the Pygame window."""
        pygame.quit()