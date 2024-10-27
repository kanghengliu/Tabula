import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os


class BoatEnv(gym.Env):
    def __init__(self, max_steps=50):
        super(BoatEnv, self).__init__()

        # Action space: 0 (move west), 1 (move east)
        self.action_space = spaces.Discrete(2)

        # Observation space: 0 (left box), 1 (right box)
        self.observation_space = spaces.Discrete(2)

        # Starting state is random
        self.state = np.random.choice([0, 1])

        # Define wind probabilities
        self.wind_prob = 0.7  # Wind blowing east (0.7) or west (0.3)

        # Define steps for an episode
        self.steps = 0
        self.max_steps = max_steps

        # Pygame setup
        pygame.init()
        self.screen_width = 400
        self.screen_height = 250
        self.box_size = 150

        # Screen setup
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Boat Environment")
        self.clock = pygame.time.Clock()

        # Dynamic path to assets folder
        current_dir = os.path.dirname(
            __file__
        )  # Get the directory of the current file (boat.py)
        asset_path = os.path.join(
            current_dir, "../assets/boat.png"
        )  # Build the full path to the boat image

        # Load the boat image
        self.boat_image = pygame.image.load(asset_path)
        self.boat_image = pygame.transform.scale(
            self.boat_image, (50, 50)
        )  # Scale it to fit

        # Font setup
        self.font = pygame.font.SysFont(None, 20)  # Default system font with size 20

        # Variables to store wind and action for display
        self.wind_direction_str = ""
        self.action_direction_str = ""

    def reset(self):
        # Reset the environment, start in a random state
        self.state = np.random.choice([0, 1])
        self.steps = 0
        info = {}  # Add an empty dictionary as the second return value
        return self.state, info

    def step(self, action):
        # Determine wind direction: 1 (east) with prob 0.7, 0 (west) with prob 0.3
        wind_direction = np.random.choice([0, 1], p=[0.3, 0.7])

        self.steps += 1

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

        terminated = False
        truncated = False
        if self.steps >= self.max_steps:  # Terminate after max steps
            terminated = True
        # Return observation (state), reward, terminated, truncated, info (empty dict)
        return self.state, reward, terminated, truncated, {}

    def render(self, mode="human"):
        # Clear the screen with a white background
        self.screen.fill((255, 255, 255))

        # Define a light blue color (RGB value for light blue is (173, 216, 230))
        light_blue = (173, 216, 230)

        # Draw filled light blue boxes
        pygame.draw.rect(
            self.screen, light_blue, pygame.Rect(50, 25, self.box_size, self.box_size)
        )  # Left box
        pygame.draw.rect(
            self.screen, light_blue, pygame.Rect(200, 25, self.box_size, self.box_size)
        )  # Right box

        # Draw black borders around the boxes
        pygame.draw.rect(
            self.screen, (0, 0, 0), pygame.Rect(50, 25, self.box_size, self.box_size), 2
        )  # Left box outline
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            pygame.Rect(200, 25, self.box_size, self.box_size),
            2,
        )  # Right box outline

        # Draw the boat in the appropriate box
        if self.state == 0:  # Boat in the left box
            self.screen.blit(self.boat_image, (100, 75))  # Boat position in left box
        else:  # Boat in the right box
            self.screen.blit(self.boat_image, (250, 75))  # Boat position in right box

        # Render wind and action direction
        wind_text = self.font.render(
            f"Wind: {self.wind_direction_str}", True, (0, 0, 0)
        )
        action_text = self.font.render(
            f"Action: {self.action_direction_str}", True, (0, 0, 0)
        )

        # Blit the text to the screen
        self.screen.blit(wind_text, (160, 200))  # Position wind text
        self.screen.blit(action_text, (160, 220))  # Position action text

        # Update the display
        pygame.display.flip()

        # # Limit the frame rate
        # self.clock.tick(10)

    def close(self):
        # Close the Pygame window
        pygame.quit()
