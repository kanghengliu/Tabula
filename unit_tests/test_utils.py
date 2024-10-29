import unittest
import numpy as np
import pygame
import os
from tabula.utils import Utils
from tabula.environments import BoatEnv
from tabula.solvers import DynamicProgramming


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = BoatEnv()
        cls.solver = DynamicProgramming(cls.env)
        cls.solver.train(episodes=10, max_steps=10)
        cls.policy = cls.solver.policy

    def test_save_and_load_policy(self):
        policy = np.array([[0.5, 0.5], [0.3, 0.7]])
        Utils.save_policy(policy, "test_policy.pkl")
        loaded_policy = Utils.load_policy("test_policy.pkl")
        np.testing.assert_array_equal(policy, loaded_policy)
        os.remove("test_policy.pkl")  # Clean up after test

    def test_epsilon_greedy(self):
        action = Utils.epsilon_greedy(
            self.env, [0.6, 0.4], 0, epsilon=0.1, is_q_values=True
        )
        self.assertIn(action, [0, 1])


if __name__ == "__main__":
    unittest.main()
