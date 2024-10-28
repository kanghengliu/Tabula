import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import os 

class BoatEnv(gym.Env):
    def __init__(self):
        super(BoatEnv, self).__init__()
        
        # Action space: 0 (move west), 1 (move east)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: 0 (left box), 1 (right box)
        self.observation_space = spaces.Discrete(2)
        
        # Starting state is random
        self.state = np.random.choice([0, 1])
        
        # Define wind probabilities
        self.wind_prob = 0.7  # Wind blowing east (0.7) or west (0.3)
        
        # Pygame setup
        pygame.init()
        self.screen_width = 400
        self.screen_height = 250
        self.box_size = 150
        
        # Screen setup
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Boat Environment')
        self.clock = pygame.time.Clock()

        # Dynamic path to assets folder
        current_dir = os.path.dirname(__file__)  # Get the directory of the current file (boat.py)
        asset_path = os.path.join(current_dir, './assets/boat.png')  # Build the full path to the boat image

        # Load the boat image
        self.boat_image = pygame.image.load(asset_path)
        self.boat_image = pygame.transform.scale(self.boat_image, (50, 50))  # Scale it to fit

        # Font setup
        self.font = pygame.font.SysFont(None, 20)  # Default system font with size 20

        # Variables to store wind and action for display
        self.wind_direction_str = ""
        self.action_direction_str = ""

    def reset(self):
        # Reset the environment, start in a random state
        self.state = np.random.choice([0, 1])

        # make info an empty dictionary
        info = {}

        return self.state, info
    
    def step(self, action):
        # Determine wind direction: 1 (east) with prob 0.7, 0 (west) with prob 0.3
        wind_direction = np.random.choice([0, 1], p=[0.3, 0.7])

        # Set wind direction for display
        self.wind_direction_str = "East" if wind_direction == 1 else "West"
        
        # Set action direction for display
        self.action_direction_str = "East" if action == 1 else "West"
        
        reward = 0  # Initialize reward
        
        # State transition logic
        if action == 1:  # Agent tries to move east
            if self.state == 0 and wind_direction == 1:  # Moves from left to right
                self.state = 1
                reward = 2  # Transition between states
            elif self.state == 1:  # Agent is in the right box
                if wind_direction == 0:  # Wind blowing west
                    self.state = 1
                    reward = 4  # Hit the right wall
                else:  # Wind blowing east
                    self.state = 1
                    reward = 3  # Stayed in right box (wind blocked movement)
            else:
                reward = 1  # Stayed in the left box (wind blocked movement)
                
        elif action == 0:  # Agent tries to move west
            if self.state == 1 and wind_direction == 0:  # Moves from right to left
                self.state = 0
                reward = 2  # Transition between states
            elif self.state == 0:  # Agent is in the left box
                if wind_direction == 1:  # Wind blowing east
                    self.state = 0
                    reward = 0  # Hit the left wall
                else:  # Wind blowing west
                    self.state = 0
                    reward = 1  # Stayed in the left box (wind blocked movement)

        terminated = False  # Termination flag
        truncated = False  # Truncation flag

        return self.state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        # Clear the screen with a white background
        self.screen.fill((255, 255, 255))
        
        # Define a light blue color (RGB value for light blue is (173, 216, 230))
        light_blue = (173, 216, 230)

        # Draw filled light blue boxes
        pygame.draw.rect(self.screen, light_blue, pygame.Rect(50, 25, self.box_size, self.box_size))  # Left box
        pygame.draw.rect(self.screen, light_blue, pygame.Rect(200, 25, self.box_size, self.box_size))  # Right box

        # Draw black borders around the boxes
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(50, 25, self.box_size, self.box_size), 2)  # Left box outline
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(200, 25, self.box_size, self.box_size), 2)  # Right box outline
        
        # Draw the boat in the appropriate box
        if self.state == 0:  # Boat in the left box
            self.screen.blit(self.boat_image, (100, 75))  # Boat position in left box
        else:  # Boat in the right box
            self.screen.blit(self.boat_image, (250, 75))  # Boat position in right box

        # Render wind and action direction
        wind_text = self.font.render(f"Wind: {self.wind_direction_str}", True, (0, 0, 0))
        action_text = self.font.render(f"Action: {self.action_direction_str}", True, (0, 0, 0))
        
        # Blit the text to the screen
        self.screen.blit(wind_text, (160, 200))   # Position wind text
        self.screen.blit(action_text, (160, 220))  # Position action text
        
        # Update the display
        pygame.display.flip()
        
        # # Limit the frame rate
        # self.clock.tick(10)

    def close(self):
        # Close the Pygame window
        pygame.quit()

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
        asset_path = os.path.join(current_dir, './assets/robot.png')  # Build the full path to the robot image

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

        # make info an empty dictionary
        info = {}

        return self.agent_pos, info

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
        terminated = self.agent_pos in self.terminal_states

        truncated = False  # Truncation flag

        return self.agent_pos, reward, terminated, truncated, {}

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
        asset_path = os.path.join(current_dir, './assets/robot.png')  # Path to the robot image

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

        # # Limit the frame rate
        # self.clock.tick(10)

    def close(self):
        """Close the Pygame window."""
        pygame.quit()