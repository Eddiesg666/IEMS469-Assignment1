"""Behavioral checks for online on-policy Pong point updates (synthetic only)."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
import pg


class SimpleActionSpace:
    def seed(self, seed):
        self.seed_value = seed


class ThreePointPong:
    """Six steps, deterministic points at steps 2, 4, 6."""
    action_space = SimpleActionSpace()

    def reset(self, seed=None):
        self.step_count = 0
        return self._obs(), {}

    def _obs(self):
        rgb = np.zeros((210, 160, 3), dtype=np.uint8)
        rgb[60 + self.step_count, 10:13, 0] = 255
        rgb[120:130, 145:148, 0] = 255
        return rgb

    def step(self, action):
        assert action in (2, 3)
        self.step_count += 1
        reward = (1.0 if self.step_count == 2 else
                  -1.0 if self.step_count == 4 else
                  1.0 if self.step_count == 6 else 0.0)
        return self._obs(), reward, self.step_count == 6, False, {}

    def close(self):
        pass


def _config(tmp_path):
    return pg.Config(game='pong', baseline='running', episodes=3,
                     batch_episodes=2, lr=1e-4, gamma=0.99, seed=17,
                     device='cpu', eval_episodes=2, train_chunk=10,
                     max_steps=0, start_fire=False, out=str(tmp_path),
                     pong_observation='stackmotion', pong_return_mode='point',
                     pong_update_mode='rally', rallies_per_update=2,
                     entropy_coef=0.001, eval_stochastic=True,
                     checkpoint_every=1)


def test_stacked_motion_includes_direction():
    p = pg.PongPolicy(input_channels=3)
    x = torch.randn(2, 3, 80, 80)
    logits = p(x)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in p.parameters())


def test_online_rally_training_updates_at_point_boundaries(monkeypatch, tmp_path):
    monkeypatch.setattr(pg, 'build_env', lambda _: ThreePointPong())
    cfg = _config(tmp_path)
    pg.train(cfg)
    metrics = json.loads((tmp_path/'metrics.json').read_text())
    # Three points/episode; groups of two, flush final point at game end.
    assert metrics['policy_updates'] == 3 * 2
    assert metrics['eval_episode_count'] == 2
    assert metrics['train_episode_count'] == 3
    assert np.isfinite(metrics['gradient_norm_mean_before_clipping'])
    assert (tmp_path/'checkpoint_latest.pt').is_file()
    assert (tmp_path/'policy.pt').is_file()
    assert len(metrics['trace_train_first10']) == 6
    assert 'frame_channel_nonzero_pixels' in metrics['trace_train_first10'][0]['state']
    assert len(metrics['trace_train_first10'][0]['state']['frame_channel_nonzero_pixels']) == 3


def test_rally_mode_rejects_episode_returns(tmp_path):
    cfg = replace(_config(tmp_path), pong_return_mode='episode')
    with pytest.raises(ValueError, match='point'):
        pg.train(cfg)


def test_regular_episode_mode_remains_default():
    assert pg.Config(game='pong', baseline='none', episodes=1, batch_episodes=1,
                     lr=0.0001, gamma=0.99, seed=0, device='cpu',
                     eval_episodes=1, train_chunk=256, max_steps=0,
                     start_fire=True, out='runs/dummy').pong_update_mode == 'episode'


def test_visual_diagnostic_generates_png_and_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(pg, 'build_env', lambda _: ThreePointPong())
    cfg = _config(tmp_path)
    pg.diagnose_pong_input(cfg)
    assert (tmp_path/'pong_input_audit.png').is_file()
    rows = json.loads((tmp_path/'pong_input_audit.json').read_text())
    assert len(rows) >= 1
    assert all(x['foreground_pixels'] > 0 for x in rows)
