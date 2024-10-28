import pickle
import pygame
import numpy as np
from PIL import Image  # Add for GIF creation
from tabula.environments import *  # Import environments


class Utils:
    @staticmethod
    def save_policy(policy, filename="optimal_policy.pkl"):
        """Saves the optimal policy to a file."""
        with open(filename, "wb") as f:
            pickle.dump(policy, f)
        print(f"Optimal policy saved to {filename}")

    @staticmethod
    def load_policy(filename="optimal_policy.pkl"):
        """Loads the optimal policy from a file."""
        with open(filename, "rb") as f:
            policy = pickle.load(f)
        print(f"Optimal policy loaded from {filename}")
        return policy

    @staticmethod
    def _state_to_index(env, state):
        """Converts a (row, col) tuple into a unique index for GridWorld or returns the state for discrete spaces."""
        if isinstance(state, tuple) and hasattr(env, "grid_width"):
            # GridWorld case: state is a (row, col) tuple, and env has grid_width
            return state[0] * env.grid_width + state[1]
        return state  # For environments with Discrete space like BoatEnv

    @staticmethod
    def epsilon_greedy(env, action_values, state, epsilon=0.1, is_q_values=False):
        """
        Returns an action using epsilon-greedy strategy, compatible with both policies and Q-values.
        If is_q_values is True, action_values is treated as Q-values; otherwise, as a policy.
        """
        if is_q_values:
            # Treat action_values as Q-values and construct an epsilon-greedy policy
            q_values = action_values  # Here, action_values is already the Q-values for this specific state
            action_probs = np.ones(env.action_space.n) * epsilon / env.action_space.n
            best_action = np.argmax(q_values)
            action_probs[best_action] += (1.0 - epsilon)
        else:
            # Treat action_values as a policy and apply epsilon-greedy using state_idx
            state_idx = Utils._state_to_index(env, state)
            policy = action_values[state_idx]
            action_probs = (1 - epsilon) * policy + (epsilon / env.action_space.n)

        # Choose an action based on the adjusted probabilities
        return np.random.choice(np.arange(env.action_space.n), p=action_probs)

    @staticmethod
    def draw_arrow(screen, x, y, direction, size=50, color=(0, 0, 0)):
        """Draws an arrow at the given (x, y) location, pointing in the given direction."""
        half_size = size // 2
        arrow_head_size = size * 0.6  # Adjust size of the arrowhead
        shaft_length = size * 0.35  # Length of the shaft

        # Calculate the start and end points of the shaft
        if direction == 0:  # Left
            start = (x + half_size, y)
            end = (x - shaft_length, y)
            arrow_tip = (x - half_size, y)
            arrow_head = [
                (arrow_tip[0], arrow_tip[1] - arrow_head_size // 2),
                (arrow_tip[0], arrow_tip[1] + arrow_head_size // 2),
                (arrow_tip[0] - arrow_head_size, arrow_tip[1]),
            ]
        elif direction == 1:  # Right
            start = (x - half_size, y)
            end = (x + shaft_length, y)
            arrow_tip = (x + half_size, y)
            arrow_head = [
                (arrow_tip[0], arrow_tip[1] - arrow_head_size // 2),
                (arrow_tip[0], arrow_tip[1] + arrow_head_size // 2),
                (arrow_tip[0] + arrow_head_size, arrow_tip[1]),
            ]
        elif direction == 2:  # Up
            start = (x, y + half_size)
            end = (x, y - shaft_length)
            arrow_tip = (x, y - half_size)
            arrow_head = [
                (arrow_tip[0] - arrow_head_size // 2, arrow_tip[1]),
                (arrow_tip[0] + arrow_head_size // 2, arrow_tip[1]),
                (arrow_tip[0], arrow_tip[1] - arrow_head_size),
            ]
        elif direction == 3:  # Down
            start = (x, y - half_size)
            end = (x, y + shaft_length)
            arrow_tip = (x, y + half_size)
            arrow_head = [
                (arrow_tip[0] - arrow_head_size // 2, arrow_tip[1]),
                (arrow_tip[0] + arrow_head_size // 2, arrow_tip[1]),
                (arrow_tip[0], arrow_tip[1] + arrow_head_size),
            ]

        # Draw the shaft of the arrow (line)
        pygame.draw.line(screen, color, start, end, 5)

        # Draw the arrowhead (triangle)
        pygame.draw.polygon(screen, color, arrow_head)

    @staticmethod
    def render_optimal_policy(
        env, policy, save_image=False, image_filename="policy_visualization.png"
    ):
        """Renders the optimal policy using arrows for BoatEnv, GridWorldEnv, and GeosearchEnv."""
        arrow_mapping = {
            0: 2,
            1: 3,
            2: 0,
            3: 1,
        }  # Mapping for directions: Left, Right, Up, Down

        # Check if the environment is GridWorldEnv or GeosearchEnv
        if isinstance(env, (GridWorldEnv, GeosearchEnv)):
            env.screen.fill(env.colors["white"])  # Clear the screen

            # Special handling for GeosearchEnv to render water and gold distributions
            if isinstance(env, GeosearchEnv):
                # Define thresholds for water and gold visibility
                water_threshold = 0.01
                gold_threshold = 0.01

                # Maximum value scaling for color adjustment
                max_water_value = np.max(env.water_distribution)
                max_gold_value = np.max(env.gold_distribution)

            for i in range(env.grid_height):
                for j in range(env.grid_width):
                    # For GeosearchEnv, handle water and gold visibility
                    if isinstance(env, GeosearchEnv):
                        water_value = env.water_distribution[i, j]
                        gold_value = env.gold_distribution[i, j]

                        # Minimum scale factor to avoid complete blackness
                        min_scale_factor = 0.4

                        if water_value > water_threshold:
                            # Darken water color based on its value (higher = darker)
                            scale_factor = max(
                                1 - (water_value / max_water_value), min_scale_factor
                            )
                            base_water_color = np.array(env.colors["water"])
                            darkened_water_color = (
                                base_water_color * scale_factor
                            ).astype(int)
                            cell_color = tuple(darkened_water_color)
                        elif gold_value > gold_threshold:
                            # Darken gold color based on its value (higher = darker)
                            scale_factor = max(
                                1 - (gold_value / max_gold_value), min_scale_factor
                            )
                            base_gold_color = np.array(env.colors["gold"])
                            darkened_gold_color = (
                                base_gold_color * scale_factor
                            ).astype(int)
                            cell_color = tuple(darkened_gold_color)
                        else:
                            # Use the pre-generated background color for each cell
                            cell_color = env.background_color_grid[i, j]
                    else:
                        # For GridWorldEnv, default to white or specific colors based on state
                        cell_color = env.colors["white"]
                        if hasattr(env, "walls") and (i, j) in env.walls:
                            cell_color = env.colors["blue"]  # Wall cells
                        if (
                            hasattr(env, "terminal_states")
                            and (i, j) in env.terminal_states
                        ):
                            cell_color = (
                                env.colors["red"]
                                if env.grid[i, j] == -50
                                else env.colors["green"]
                            )

                    # Draw the cell
                    pygame.draw.rect(
                        env.screen,
                        cell_color,
                        pygame.Rect(
                            j * env.cell_size,
                            i * env.cell_size,
                            env.cell_size,
                            env.cell_size,
                        ),
                    )
                    pygame.draw.rect(
                        env.screen,
                        env.colors["black"],
                        pygame.Rect(
                            j * env.cell_size,
                            i * env.cell_size,
                            env.cell_size,
                            env.cell_size,
                        ),
                        1,
                    )

                    # Determine if we should draw an arrow
                    draw_arrow = True
                    if isinstance(env, GridWorldEnv):
                        # Skip arrows in wall or terminal states for GridWorldEnv
                        if (hasattr(env, "walls") and (i, j) in env.walls) or (
                            hasattr(env, "terminal_states")
                            and (i, j) in env.terminal_states
                        ):
                            draw_arrow = False

                    # Draw arrows based on the optimal policy if allowed
                    if draw_arrow:
                        state_idx = i * env.grid_width + j
                        if state_idx < len(policy):
                            optimal_action = np.argmax(policy[state_idx])
                            direction = arrow_mapping[optimal_action]
                            Utils.draw_arrow(
                                env.screen,
                                j * env.cell_size + env.cell_size // 2,
                                i * env.cell_size + env.cell_size // 2,
                                direction,
                                size=env.cell_size * 0.4,
                            )
            if save_image:
                Utils.save_image(env.screen, filename=image_filename)
            pygame.display.flip()

        # For BoatEnv
        elif isinstance(env, BoatEnv):
            env.screen.fill((255, 255, 255))  # White background

            box_size = 150
            pygame.draw.rect(
                env.screen, (173, 216, 230), pygame.Rect(50, 25, box_size, box_size)
            )
            pygame.draw.rect(
                env.screen, (173, 216, 230), pygame.Rect(200, 25, box_size, box_size)
            )
            pygame.draw.rect(
                env.screen, (0, 0, 0), pygame.Rect(50, 25, box_size, box_size), 2
            )
            pygame.draw.rect(
                env.screen, (0, 0, 0), pygame.Rect(200, 25, box_size, box_size), 2
            )

            for state in range(env.observation_space.n):
                optimal_action = np.argmax(policy[state])

                if state == 0:
                    center_pos = (50 + box_size // 2, 25 + box_size // 2)
                    Utils.draw_arrow(
                        env.screen,
                        center_pos[0],
                        center_pos[1],
                        optimal_action,
                        size=50,
                        color=(0, 0, 0),
                    )
                elif state == 1:
                    center_pos = (200 + box_size // 2, 25 + box_size // 2)
                    Utils.draw_arrow(
                        env.screen,
                        center_pos[0],
                        center_pos[1],
                        optimal_action,
                        size=50,
                        color=(0, 0, 0),
                    )

            pygame.display.flip()

    @staticmethod
    def create_gif(frames, filename="gameplay.gif", duration=100):
        """Creates a GIF from a list of Pygame frames."""
        pil_images = [Image.fromarray(frame) for frame in frames]
        pil_images[0].save(
            filename,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=0,
        )
        print(f"Gameplay GIF saved as {filename}")

    @staticmethod
    def save_image(screen, filename="policy_visualization.png"):
        """Saves the current Pygame screen as an image."""
        frame = pygame.surfarray.array3d(screen)
        image = Image.fromarray(
            np.transpose(frame, (1, 0, 2))
        )  # Convert for Pillow compatibility
        image.save(filename)
        print(f"Optimal policy visualization saved as {filename}")

    @staticmethod
    def run_optimal_policy(
        env,
        policy,
        episodes=5,
        max_steps=50,
        delay_ms=100,
        save_gif=False,
        gif_filename="gameplay.gif",
    ):
        """Simulate episodes following the optimal policy, render, and optionally save as GIF."""
        frames = []
        total_reward = 0  # Track total reward across all episodes

        for episode in range(episodes):
            state, info = env.reset()  # Reset environment at the start of each episode
            terminated = False
            episode_reward = 0  # Track reward for this episode
            steps = 0

            while not terminated:
                # Render the environment and capture the frame if saving GIF
                env.render()
                if save_gif:
                    frame = pygame.surfarray.array3d(
                        env.screen
                    )  # Capture the screen as a 3D array
                    frames.append(
                        np.transpose(frame, (1, 0, 2))
                    )  # Convert for Pillow compatibility

                # Introduce a delay so actions can be observed (delay in milliseconds)
                pygame.time.wait(delay_ms)

                # Take the best action according to the optimal policy
                action = np.argmax(policy[Utils._state_to_index(env, state)])

                # Step in the environment using the optimal action
                next_state, reward, terminated, _, _ = env.step(action)

                # Update the state and accumulate reward
                state = next_state
                episode_reward += reward  # Accumulate the episode's reward

                # Increment step count
                steps += 1

                # End the episode after reaching the step limit
                if steps >= max_steps:
                    terminated = True

            total_reward += episode_reward  # Add episode reward to total reward

        # Save GIF if required
        if save_gif:
            Utils.create_gif(frames, filename=gif_filename, duration=delay_ms)

        avg_reward = (
            total_reward / episodes
        )  # Calculate average reward across all episodes
        print(f"Average reward following optimal policy: {avg_reward:.2f}")

