"""New tests for rally-segmented Pong policy gradient returns."""
from dataclasses import replace
import numpy as np
import torch
from pg import Config, discounted_returns, pong_point_returns, returns_for_episode, update_from_batch, PongPolicy


def test_pong_rally_returns_do_not_cross_scored_points():
    rewards = [0, 0, +1, 0, 0, -1]
    got = pong_point_returns(rewards, gamma=0.9)
    expected = np.array([0.81, 0.9, 1, -0.81, -0.9, -1], dtype=np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    assert discounted_returns(rewards, 0.9)[0] != got[0]


def test_point_mode_is_opt_in_and_not_used_for_cartpole():
    cfg = Config(game='pong', baseline='none', episodes=1, batch_episodes=1,
                 lr=1e-4, gamma=0.9, seed=1, device='cpu', eval_episodes=1,
                 train_chunk=10, max_steps=0, start_fire=True, out='unused',
                 pong_return_mode='episode')
    rewards = [0, 1, 0, -1]
    np.testing.assert_allclose(returns_for_episode(rewards, cfg),
                               discounted_returns(rewards, cfg.gamma))
    np.testing.assert_allclose(returns_for_episode(rewards, replace(cfg, pong_return_mode='point')),
                               pong_point_returns(rewards, cfg.gamma))
    np.testing.assert_allclose(returns_for_episode(rewards, replace(cfg, game='cartpole', pong_return_mode='point')),
                               discounted_returns(rewards, cfg.gamma))


def test_point_mode_produces_finite_nonzero_policy_gradient():
    torch.manual_seed(2)
    model = PongPolicy(input_channels=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    cfg = Config(game='pong', baseline='running', episodes=1, batch_episodes=1,
                 lr=1e-4, gamma=0.99, seed=2, device='cpu', eval_episodes=1,
                 train_chunk=4, max_steps=0, start_fire=True, out='unused',
                 pong_observation='stack', pong_return_mode='point')
    rng = np.random.default_rng(2)
    batch = [{'states': rng.integers(0,2,size=(6,2,80,80), dtype=np.uint8),
              'actions': np.array([0,1,0,1,0,1]),
              'rewards': [0,0,1,0,0,-1]}]
    before = [p.detach().clone() for p in model.parameters()]
    baseline, norm = update_from_batch(model, opt, batch, cfg, torch.device('cpu'), -0.2)
    assert np.isfinite(baseline) and np.isfinite(norm) and norm > 0
    assert any(not torch.equal(p0, p1) for p0,p1 in zip(before, model.parameters()))
