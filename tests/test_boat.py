import sys
import os

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tabula.games import boat  # Import your custom environment
import numpy as np
import pygame  # Import pygame to handle events like quitting

# Create the custom BoatEnv environment
env = boat.BoatEnv()

# Reset the environment to initialize it
state = env.reset()

# Run the environment for a certain number of steps
num_steps = 1000  # Number of simulation steps
for step in range(num_steps):
    print(f"Step {step + 1}:")
    
    # Check for quitting the Pygame window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()
    
    # Choose an action (randomly for now, 0 = west, 1 = east)
    action = env.action_space.sample()
    
    # Take a step in the environment
    new_state, reward, done, info = env.step(action)
    
    # Print what happened during this step
    print(f"Action taken: {'East' if action == 1 else 'West'}, New State: {new_state}, Reward: {reward}")
    
    # Render the current state (with Pygame)
    env.render()

# Close the environment rendering window
env.close()