# ./unit_tests/test_solvers.py
import unittest
from tabula.environments import BoatEnv
from tabula.solvers import DynamicProgramming, MonteCarloES, TemporalDifference
import numpy as np


class TestDynamicProgramming(unittest.TestCase):
    def setUp(self):
        self.env = BoatEnv()
        self.solver = DynamicProgramming(self.env)

    def test_policy_shape(self):
        self.assertEqual(
            self.solver.policy.shape, (self.solver.num_states, self.env.action_space.n)
        )

    def test_value_iteration(self):
        self.solver.compute_transition_model()  # Ensure transition_model is set before calling value_iteration
        policy = self.solver.value_iteration()
        self.assertIsInstance(policy, np.ndarray)


class TestMonteCarloES(unittest.TestCase):
    def setUp(self):
        self.env = BoatEnv()
        self.solver = MonteCarloES(self.env)

    def test_q_table_initialization(self):
        self.assertEqual(
            self.solver.Q.shape, (self.solver.state_size, self.env.action_space.n)
        )

    def test_training(self):
        policy = self.solver.train(episodes=10)
        self.assertIsInstance(policy, np.ndarray)


class TestTemporalDifference(unittest.TestCase):
    def setUp(self):
        self.env = BoatEnv()
        self.solver = TemporalDifference(self.env)

    def test_q_table_initialization(self):
        self.assertEqual(
            self.solver.q_table.shape, (self.solver.num_states, self.env.action_space.n)
        )

    def test_training(self):
        policy = self.solver.train(episodes=10)
        self.assertIsInstance(policy, np.ndarray)


if __name__ == "__main__":
    unittest.main()
