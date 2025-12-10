import unittest
import numpy as np
import pooltool as pt
from agent import NewAgent
from pooltool.objects import Ball, Table
from unittest.mock import MagicMock, patch

class TestNewAgent(unittest.TestCase):
    def setUp(self):
        self.agent = NewAgent()
        self.ball_radius = 0.028575

    def test_calculate_ghost_ball_pos_straight(self):
        # Target at (1, 0), Pocket at (2, 0)
        # Vector target->pocket is (1, 0)
        # Ghost ball should be at Target - 2R * (1, 0) = (1 - 2R, 0)
        target_pos = [1.0, 0.0, 0.0]
        pocket_pos = [2.0, 0.0, 0.0]
        
        ghost_pos = self.agent._calculate_ghost_ball_pos(target_pos, pocket_pos, self.ball_radius)
        
        expected_x = 1.0 - 2 * self.ball_radius
        self.assertAlmostEqual(ghost_pos[0], expected_x)
        self.assertAlmostEqual(ghost_pos[1], 0.0)
        self.assertAlmostEqual(ghost_pos[2], 0.0)

    def test_calculate_ghost_ball_pos_angle(self):
        # Target at (0, 0), Pocket at (1, 1)
        # Vector is (1, 1), normalized is (1/sqrt(2), 1/sqrt(2))
        target_pos = [0.0, 0.0, 0.0]
        pocket_pos = [1.0, 1.0, 0.0]
        
        ghost_pos = self.agent._calculate_ghost_ball_pos(target_pos, pocket_pos, self.ball_radius)
        
        dist = np.sqrt(2)
        unit_vec = np.array([1/dist, 1/dist, 0])
        expected = np.array(target_pos) - 2 * self.ball_radius * unit_vec
        
        np.testing.assert_array_almost_equal(ghost_pos, expected)

    def test_check_collision_path_clear(self):
        # Path from (0,0) to (10,0)
        # Obstacle at (5, 5) -> Far away
        start_pos = [0.0, 0.0, 0.0]
        end_pos = [10.0, 0.0, 0.0]
        
        # Create a dummy ball
        obs_ball = MagicMock()
        obs_ball.state.rvw = [[5.0, 5.0, 0.0], [0,0,0]]
        
        collision = self.agent._check_collision_path(start_pos, end_pos, [obs_ball], self.ball_radius)
        self.assertFalse(collision)

    def test_check_collision_path_blocked(self):
        # Path from (0,0) to (10,0)
        # Obstacle at (5, 0.01) -> Very close, should collide
        start_pos = [0.0, 0.0, 0.0]
        end_pos = [10.0, 0.0, 0.0]
        
        obs_ball = MagicMock()
        obs_ball.state.rvw = [[5.0, 0.01, 0.0], [0,0,0]]
        
        collision = self.agent._check_collision_path(start_pos, end_pos, [obs_ball], self.ball_radius)
        self.assertTrue(collision)

    def test_check_collision_path_behind(self):
        # Path from (0,0) to (10,0)
        # Obstacle at (-2, 0) -> Behind start point
        start_pos = [0.0, 0.0, 0.0]
        end_pos = [10.0, 0.0, 0.0]
        
        obs_ball = MagicMock()
        obs_ball.state.rvw = [[-2.0, 0.0, 0.0], [0,0,0]]
        
        collision = self.agent._check_collision_path(start_pos, end_pos, [obs_ball], self.ball_radius)
        self.assertFalse(collision)

    @patch('agent.pt.simulate')
    @patch('agent.analyze_shot_for_reward')
    def test_decision_random_fallback(self, mock_analyze, mock_simulate):
        # If balls is None
        action = self.agent.decision(balls=None)
        self.assertIn('V0', action)
        self.assertIn('phi', action)

    @patch('agent.pt.simulate')
    def test_decision_no_valid_targets(self, mock_simulate):
        # Setup mocks
        balls = {}
        # Cue ball
        cue_ball = MagicMock()
        cue_ball.state.rvw = [[0, 0, 0], [0,0,0]]
        cue_ball.params.R = self.ball_radius
        balls['cue'] = cue_ball
        
        # Target ball 1 (pocketed)
        ball_1 = MagicMock()
        ball_1.state.s = 4 # Pocketed
        balls['1'] = ball_1
        
        my_targets = ['1']
        table = MagicMock()
        
        # Should fallback to random or 8 ball?
        # If '1' is pocketed, targets becomes ['8']
        # If '8' is not in balls (or pocketed?), logic might fail or fallback
        # Let's add '8' to balls
        ball_8 = MagicMock()
        ball_8.state.s = 1 # On table
        ball_8.state.rvw = [[1, 1, 0], [0,0,0]]
        balls['8'] = ball_8
        
        # Table pockets
        pocket = MagicMock()
        pocket.center = [2, 2, 0]
        table.pockets = {'p1': pocket}
        
        # Mock calculate_ghost_ball_pos to return valid pos
        with patch.object(self.agent, '_calculate_ghost_ball_pos', return_value=np.array([0.9, 0.9, 0])):
             # Mock check_collision to return False (no collision)
            with patch.object(self.agent, '_check_collision_path', return_value=False):
                action = self.agent.decision(balls, my_targets, table)
                
        self.assertIsNotNone(action)
        self.assertIn('V0', action)

if __name__ == '__main__':
    unittest.main()
