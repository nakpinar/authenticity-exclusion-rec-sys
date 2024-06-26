"""
Test the methods in main_util.py
"""

import unittest
import numpy as np

from main_util import *

class TestModel(unittest.TestCase):

    # Test Model.init
    def test_agent_groups(self):
        sim = Model()
        groups = np.array([agent.group for agent in sim.agents])
        groups, counts = np.unique(groups, return_counts=True)
        assert (np.abs(counts / counts.sum() - sim.group_shares) < 0.1).all()

    def test_agent_preferences(self):
        sim = Model()
        groups = np.array([agent.group for agent in sim.agents])
        preferences = np.array([agent.preferences for agent in sim.agents])
        preferences0 = preferences[groups == 0].mean(axis=0)
        preferences1 = preferences[groups == 1].mean(axis=0)
        assert (np.abs(np.array([preferences0,preferences1]) - sim.preference_means / sim.preference_means.sum(axis=1)[:,np.newaxis]) < 0.1).all()

    def test_edges_complete(self):
        sim = Model(init_edges='complete')
        assert (sim.adj_matrix == np.ones((sim.num_agents,sim.num_agents))).all()

    def test_edges_random(self):
        sim = Model(init_edges='random')
        assert np.abs(sim.adj_matrix.mean() - 0.5) < 0.01

    def test_edges_homophilic(self):
        sim = Model(init_edges='homophilic')
        groups = np.array([agent.group for agent in sim.agents])
        assert np.abs(sim.adj_matrix[groups == 0, groups == 0].mean() - 0.6) < 0.05
        assert np.abs(sim.adj_matrix[groups == 1, groups == 1].mean() - 0.6) < 0.05
        assert np.abs(sim.adj_matrix[groups == 0, :][:, groups == 1].mean() - 0.3) < 0.05
        assert np.abs(sim.adj_matrix[groups == 1, :][:, groups == 0].mean() - 0.3) < 0.05

    # Test Model.content_step
    def test_content_step(self):
        sim = Model()
        content = []

        # Collect 100 time steps worth of content
        for _ in range(100):
            sim.content_step()
            content.extend(sim.content[0])

        # Assert right number of items created, content_prob * num_agents * 100
        assert np.abs(len(content) / (0.2 * sim.num_agents * 100) - 1) < 0.1

        # Assert topic distribution is good by group
        groups  = np.array([item.creator.group for item in content])
        topics = np.array([item.topic for item in content])
        _, counts = np.unique(topics[groups == 0], return_counts=True)
        assert (np.abs(counts / counts.sum() - sim.preference_means[0] / sim.preference_means[0].sum()) < 0.01).all()
        _, counts = np.unique(topics[groups == 1], return_counts=True)
        assert (np.abs(counts / counts.sum() - sim.preference_means[1] / sim.preference_means[1].sum()) < 0.01).all()


if __name__ == '__main__':
    unittest.main()