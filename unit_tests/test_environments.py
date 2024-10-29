import unittest
from tabula.environments import BoatEnv, GridWorldEnv, GeosearchEnv
import numpy as np


class TestBoatEnv(unittest.TestCase):
    def setUp(self):
        self.env = BoatEnv()

    def test_initial_state(self):
        state, _ = self.env.reset()
        self.assertIn(state, [0, 1])

    def test_step_function(self):
        state, reward, terminated, truncated, _ = self.env.step(0)
        self.assertIn(state, [0, 1])
        self.assertIsInstance(reward, int)

    def tearDown(self):
        self.env.close()


class TestGridWorldEnv(unittest.TestCase):
    def setUp(self):
        self.env = GridWorldEnv()

    def test_initial_position(self):
        pos, _ = self.env.reset()
        self.assertEqual(pos, (0, 0))

    def test_step(self):
        pos, reward, terminated, truncated, _ = self.env.step(1)
        self.assertTrue(
            isinstance(reward, (int, np.integer))
        )  # Updated to accept np.integer

    def tearDown(self):
        self.env.close()


class TestGeosearchEnv(unittest.TestCase):
    def setUp(self):
        self.env = GeosearchEnv()

    def test_reset(self):
        pos, _ = self.env.reset()
        self.assertIsInstance(pos, tuple)

    def test_step(self):
        self.env.reset()  # Ensure agent_pos is initialized
        pos, reward, terminated, truncated, _ = self.env.step(1)
        self.assertIsInstance(reward, float)

    def tearDown(self):
        self.env.close()


if __name__ == "__main__":
    unittest.main()
