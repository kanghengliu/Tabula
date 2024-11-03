import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import os 

class BoatEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(BoatEnv, self).__init__()
        
        # Action space: 0 (move west), 1 (move east)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: 0 (left box), 1 (right box)
        self.observation_space = spaces.Discrete(2)
        
        # Starting state is random
        self.state = np.random.choice([0, 1])
        
        # Define wind probabilities
        self.wind_prob = 0.7  # Wind blowing east (0.7) or west (0.3)
        
        # Pygame setup only if rendering is enabled
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.boat_image = None
        self.font = None
        
        # Variables to store wind and action for display
        self.wind_direction_str = ""
        self.action_direction_str = ""

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

    def reset(self):
        self.state = np.random.choice([0, 1])
        return self.state, {}

    def _init_render(self):
        """Initialize Pygame components if not already initialized"""
        if self.screen is None and self.render_mode is not None:
            pygame.init()
            self.screen_width = 400
            self.screen_height = 250
            self.box_size = 150
            
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Boat Environment')
            self.clock = pygame.time.Clock()

            # Load the boat image
            current_dir = os.path.dirname(__file__)
            asset_path = os.path.join(current_dir, './assets/boat.png')
            self.boat_image = pygame.image.load(asset_path)
            self.boat_image = pygame.transform.scale(self.boat_image, (50, 50))

            # Font setup
            self.font = pygame.font.SysFont(None, 20)

    def render(self):
        if self.render_mode is None:
            return
            
        self._init_render()
        
        # Rest of the render code remains the same
        self.screen.fill((255, 255, 255))
        light_blue = (173, 216, 230)
        
        # Draw filled light blue boxes
        pygame.draw.rect(self.screen, light_blue, pygame.Rect(50, 25, self.box_size, self.box_size))
        pygame.draw.rect(self.screen, light_blue, pygame.Rect(200, 25, self.box_size, self.box_size))
        
        # Draw black borders
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(50, 25, self.box_size, self.box_size), 2)
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(200, 25, self.box_size, self.box_size), 2)
        
        # Draw boat
        if self.state == 0:
            self.screen.blit(self.boat_image, (100, 75))
        else:
            self.screen.blit(self.boat_image, (250, 75))
        
        # Render text
        wind_text = self.font.render(f"Wind: {self.wind_direction_str}", True, (0, 0, 0))
        action_text = self.font.render(f"Action: {self.action_direction_str}", True, (0, 0, 0))
        self.screen.blit(wind_text, (160, 200))
        self.screen.blit(action_text, (160, 220))
        
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.boat_image = None
            self.font = None

class GridWorldEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(GridWorldEnv, self).__init__()

        # Set up grid dimensions
        self.grid_height = 6
        self.grid_width = 6

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.grid_height), spaces.Discrete(self.grid_width)))

        # Grid setup
        self.grid = np.full((self.grid_height, self.grid_width), -1)
        self.grid[1, 4] = -50
        self.grid[5, 0] = -50
        self.grid[5, 5] = 100

        # States and walls
        self.terminal_states = [(1, 4), (5, 0), (5, 5)]
        self.walls = [(0, 2), (1, 2), (3, 2), (3, 3), (3, 4), (4,2), (5, 2)]
        self.start_pos = (0, 0)
        self.agent_pos = self.start_pos

        # Pygame setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.robot_image = None
        
        # Colors
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'gray': (128, 128, 128),
        }

    def step(self, action):
        """Apply the action, return the next state, reward, done, and info."""
        if self.agent_pos in self.terminal_states:
            return self.agent_pos, 0, True, {}

        # Stochastic action: 25% chance of doing a random action
        if random.random() < 0.25:
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
        truncated = False

        return self.agent_pos, reward, terminated, truncated, {}

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos, {}

    def _init_render(self):
        """Initialize Pygame components if not already initialized"""
        if self.screen is None and self.render_mode is not None:
            pygame.init()
            self.screen_width = 600
            self.screen_height = 600
            self.cell_size = 100
            
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("GridWorld Environment")
            self.clock = pygame.time.Clock()

            # Load robot image
            current_dir = os.path.dirname(__file__)
            asset_path = os.path.join(current_dir, './assets/robot.png')
            self.robot_image = pygame.image.load(asset_path)
            self.robot_image = pygame.transform.scale(self.robot_image, (50, 50))
            
            self.font = pygame.font.SysFont(None, 24)

    def render(self):
        if self.render_mode is None:
            return
            
        self._init_render()
        
        # Rest of the render implementation remains the same
        self.screen.fill(self.colors['white'])

        for i in range(self.grid_height):
            for j in range(self.grid_width):
                cell_color = self.colors['white']

                if (i, j) in self.walls:
                    cell_color = self.colors['blue']
                elif (i, j) == (1, 4) or (i, j) == (5, 0):
                    cell_color = self.colors['red']
                elif (i, j) == (5, 5):
                    cell_color = self.colors['green']

                pygame.draw.rect(self.screen, cell_color,
                               pygame.Rect(j * self.cell_size, i * self.cell_size,
                                         self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.colors['black'],
                               pygame.Rect(j * self.cell_size, i * self.cell_size,
                                         self.cell_size, self.cell_size), 1)

        self.screen.blit(self.robot_image,
                        (self.agent_pos[1] * self.cell_size + self.cell_size // 4,
                         self.agent_pos[0] * self.cell_size + self.cell_size // 4))

        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.robot_image = None

class GeosearchEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(GeosearchEnv, self).__init__()

        # Grid setup
        self.grid_height = 25
        self.grid_width = 25
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.grid_height), spaces.Discrete(self.grid_width)))

        # Gaussian parameters
        self.mu_water = np.array([20, 20])
        self.sigma_water = np.array([[1, 0.25], [0.25, 1]])
        self.mu_gold = np.array([10, 10])
        self.sigma_gold = np.array([[1, -0.25], [-0.25, 1]])
        self.A = 0.75

        # Resource distributions
        self.water_distribution = self._create_distribution(self.mu_water, self.sigma_water)
        self.gold_distribution = self._create_distribution(self.mu_gold, self.sigma_gold)
        self.reward_matrix = self.A * self.water_distribution + (1 - self.A) * self.gold_distribution
        self.agent_pos = None

        # Pygame setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.robot_image = None
        
        # Colors and background
        self.background_colors = ['#489030', '#5d924d', '#789030', '#a5803d']
        self.background_color_grid = self._generate_background_grid()
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'water': (15, 94, 156),
            'gold': (255, 207, 64)
        }

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

        # No terminal states in this environment
        terminated = False
        truncated = False

        return self.agent_pos, reward, terminated, truncated, {}

    def reset(self):
        """Reset the environment to the initial state."""
        self.agent_pos = (random.randint(0, self.grid_height - 1), 
                         random.randint(0, self.grid_width - 1))
        return self.agent_pos, {}

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
        
        n = 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
        inv_sigma = np.linalg.inv(sigma)
        diff = pos - mu
        exponent = -0.5 * np.einsum('...k,kl,...l->...', diff, inv_sigma, diff)
        
        return n * np.exp(exponent)

    def _init_render(self):
        """Initialize Pygame components if not already initialized"""
        if self.screen is None and self.render_mode is not None:
            pygame.init()
            self.screen_width = 875
            self.screen_height = 875
            self.cell_size = 35
            
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Geosearch Environment")
            self.clock = pygame.time.Clock()

            # Load robot image
            current_dir = os.path.dirname(__file__)
            asset_path = os.path.join(current_dir, './assets/robot.png')
            self.robot_image = pygame.image.load(asset_path)
            self.robot_image = pygame.transform.scale(self.robot_image, (20, 20))

    def render(self):
        if self.render_mode is None:
            return
            
        self._init_render()
        
        # Rest of the render implementation remains the same
        self.screen.fill(self.colors['white'])

        water_threshold = 0.01
        gold_threshold = 0.01
        max_water_value = np.max(self.water_distribution)
        max_gold_value = np.max(self.gold_distribution)
        min_scale_factor = 0.4

        for i in range(self.grid_height):
            for j in range(self.grid_width):
                water_value = self.water_distribution[i, j]
                gold_value = self.gold_distribution[i, j]

                if water_value > water_threshold:
                    scale_factor = max(1 - (water_value / max_water_value), min_scale_factor)
                    base_water_color = np.array(self.colors['water'])
                    darkened_water_color = (base_water_color * scale_factor).astype(int)
                    cell_color = tuple(darkened_water_color)
                elif gold_value > gold_threshold:
                    scale_factor = max(1 - (gold_value / max_gold_value), min_scale_factor)
                    base_gold_color = np.array(self.colors['gold'])
                    darkened_gold_color = (base_gold_color * scale_factor).astype(int)
                    cell_color = tuple(darkened_gold_color)
                else:
                    cell_color = self.background_color_grid[i, j]

                pygame.draw.rect(self.screen, cell_color,
                               pygame.Rect(j * self.cell_size, i * self.cell_size,
                                         self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.colors['black'],
                               pygame.Rect(j * self.cell_size, i * self.cell_size,
                                         self.cell_size, self.cell_size), 1)

        self.screen.blit(self.robot_image,
                        (self.agent_pos[1] * self.cell_size + self.cell_size // 4,
                         self.agent_pos[0] * self.cell_size + self.cell_size // 4))

        pygame.display.flip()

    def close(self):
        if self.screen is None:
            return
        pygame.quit()
        self.screen = None
        self.clock = None
        self.robot_image = None