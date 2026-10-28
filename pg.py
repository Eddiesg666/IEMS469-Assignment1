"""REINFORCE for CartPole-v1 and ALE/Pong-v5 (with optional running baseline).

The only learned Pong actions are ALE indices 2 and 3.  An optional fixed
FIRE action at episode start serves the ball (it is NOT a policy decision).

Pong supports either binary frame *differences* [1,80,80] or two consecutive
binary frames [2,80,80]. Both use the assignment's image preprocessing.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class Config:
    game: str
    baseline: str
    episodes: int
    batch_episodes: int
    lr: float
    gamma: float
    seed: int
    device: str
    eval_episodes: int
    train_chunk: int
    max_steps: int
    start_fire: bool
    out: str
    pong_observation: str = 'difference'
    entropy_coef: float = 0.0
    eval_stochastic: bool = False
    checkpoint_every: int = 0
    resume_from: str | None = None
    pong_return_mode: str = 'episode'
    pong_update_mode: str = 'episode'
    rallies_per_update: int = 5
    pong_network: str = 'classic'
    pong_guide_coef: float = 0.0
    action2_is_down: bool = False


def build_env(game: str):
    try:
        import gymnasium as gym
    except ImportError as e:
        raise RuntimeError('Install requirements first: python -m pip install -r requirements.txt') from e
    if game == 'cartpole':
        return gym.make('CartPole-v1')
    if game == 'pong':
        try:
            import ale_py
        except ImportError as e:
            raise RuntimeError('Install Atari dependency: python -m pip install ale-py') from e
        gym.register_envs(ale_py)
        return gym.make('ALE/Pong-v5')
    raise ValueError(f'Unexpected game {game}')


def preprocess(image: np.ndarray) -> np.ndarray:
    """Assignment's 210x160x3 -> binary 80x80 preprocessing.

    A copy is necessary to avoid altering Gymnasium's frame buffer.
    """
    im = np.array(image[35:195:2, ::2, 0], copy=True)
    im[(im == 144) | (im == 109)] = 0
    im[im != 0] = 1
    result = im.astype(np.float32)
    if result.shape != (80, 80):
        raise ValueError(f'Expected Pong RGB 210x160x3; got {image.shape}')
    return result


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """G[t] = r[t] + gamma*r[t+1] + ... , without crossing game episodes."""
    values = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for i in range(len(rewards)-1, -1, -1):
        g = float(rewards[i]) + gamma * g
        values[i] = g
    return values


def pong_point_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """Discount rewards within *one Pong point*, not across point boundaries.

    Each nonzero reward signals a scored point (a natural rally boundary).
    The point result is attributed to actions in its own rally only.
    This is an optional Pong-specific credit-assignment variant, NOT the
    original full-episode discounted-return objective.
    """
    values = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        if rewards[i] != 0:
            g = 0.0
        g = float(rewards[i]) + gamma * g
        values[i] = g
    return values


def returns_for_episode(rewards: list[float], cfg: Config) -> np.ndarray:
    if cfg.game == 'pong' and cfg.pong_return_mode == 'point':
        return pong_point_returns(rewards, cfg.gamma)
    return discounted_returns(rewards, cfg.gamma)


class CartPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 128), nn.ReLU(),
                                 nn.Linear(128, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PongPolicy(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(32*8*8, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def visual_geometry(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Read paddle/ball positions from the *input frames*, without action labels.

    Pong's binary 80x80 crop has the left paddle at x~7, the player's
    right paddle at x~73, and the moving ball between them.  Background
    and scoreboard were removed by preprocess().  Normalize coordinates
    to [-1, 1]. Flags distinguish absent objects from centered objects.
    
    x: [B,C,80,80] where C>=2, (previous,current,optional motion).
    """
    if x.ndim != 4 or x.shape[-2:] != (80, 80) or x.shape[1] < 2:
        raise ValueError('Geometry policy requires 2+ 80x80 frames')
    prev = x[:, 0]
    curr = x[:, 1]
    dtype = curr.dtype
    rows = torch.arange(80, device=x.device, dtype=dtype).view(1, 80, 1)
    cols = torch.arange(80, device=x.device, dtype=dtype).view(1, 1, 80)

    def centroid(frame: torch.Tensor, xmin: int, xmax: int):
        occupancy = (frame[:, :, xmin:xmax] > 0).to(dtype)
        total = occupancy.sum(dim=(1, 2))
        yy = (occupancy * rows).sum(dim=(1, 2)) / total.clamp_min(1.0)
        xx = (occupancy * cols[:, :, xmin:xmax]).sum(dim=(1, 2)) / total.clamp_min(1.0)
        # Missing objects are represented by 0 coordinates + a 0 flag.
        present = (total > 0).to(dtype)
        yy = torch.where(total > 0, 2.0 * (yy / 79.0) - 1.0, torch.zeros_like(yy))
        xx = torch.where(total > 0, 2.0 * (xx / 79.0) - 1.0, torch.zeros_like(xx))
        return xx, yy, present

    # Left/right paddle columns are outside the central ball search region.
    _, py, p_ok = centroid(curr, 69, 79)
    _, oy, o_ok = centroid(curr, 3, 12)
    bx, by, b_ok = centroid(curr, 12, 69)
    pbx, pby, pb_ok = centroid(prev, 12, 69)
    both = b_ok * pb_ok
    velocity_x = (bx - pbx) * both
    velocity_y = (by - pby) * both
    diff = (by - py) * b_ok * p_ok
    features = torch.stack((py, oy, bx, by, velocity_x, velocity_y,
                            diff, p_ok, o_ok, b_ok, pb_ok, both), dim=1)
    return features, (b_ok * p_ok)


class PongGeometryPolicy(nn.Module):
    """Object-aware network receiving the supplied frames, not game RAM.

    A geometry branch keeps individual tiny ball pixels measurable; a
    shallow stride-1 convolution learns complementary local appearance.
    There is no hard-coded policy or access to rewards in forward().
    """
    def __init__(self, input_channels: int = 3):
        super().__init__()
        if input_channels < 2:
            raise ValueError('Use --pong-observation stack or stackmotion')
        self.local = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 12, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveMaxPool2d((4, 4)), nn.Flatten())
        self.net = nn.Sequential(nn.Linear(12*4*4 + 12, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        geo, _ = visual_geometry(x)
        img = self.local(x)
        return self.net(torch.cat((img, geo), dim=1))


def make_policy(game: str, pong_observation='difference', pong_network='classic') -> nn.Module:
    if game == 'cartpole':
        return CartPolicy()
    channels = {'difference': 1, 'stack': 2, 'stackmotion': 3}[pong_observation]
    if pong_network == 'geometry':
        return PongGeometryPolicy(channels)
    if pong_network != 'classic':
        raise ValueError(f'Unknown Pong network: {pong_network}')
    return PongPolicy(input_channels=channels)


def choose_device(name: str) -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def right_paddle_center(rgb: np.ndarray) -> float | None:
    im = preprocess(rgb)
    yy = np.where(im[:, 69:79] > 0)[0]
    return float(np.mean(yy)) if len(yy) else None


def calibrate_action2_direction(env, seed: int, start_fire: bool) -> bool:
    """Compare identical seeded trajectories under sustained action 2/3.

    Avoid guessing whether ALE's RIGHT moves up/down in image coordinates.
    Returns True if action 2 increases the paddle y coordinate (moves down).
    """
    centers = []
    for action in (2, 3):
        obs, _ = env.reset(seed=seed + 777)
        if start_fire:
            obs, *_ = env.step(1)
        for _ in range(10):
            obs, *_ = env.step(action)
        centers.append(right_paddle_center(obs))
    if any(x is None for x in centers):
        raise RuntimeError('Could not locate player paddle for action calibration')
    if abs(centers[0] - centers[1]) < 0.5:
        raise RuntimeError('Action 2 and 3 did not move the paddle in opposite directions')
    return centers[0] > centers[1]


def rollout(env, policy: nn.Module, cfg: Config, episode_seed: int,
            device: torch.device, deterministic: bool, trace_limit: int = 0,
            on_point=None):
    """Store states/actions/rewards, not graphs, so long Pong games fit in RAM."""
    obs, _info = env.reset(seed=episode_seed)
    env.action_space.seed(episode_seed)
    if cfg.game == 'pong':
        # A fixed 'FIRE' is a startup procedure, NOT one of the two learned actions.
        # Disable this with --no-start-fire if the class prohibits all FIRE steps.
        if cfg.start_fire:
            obs, fire_reward, fire_done, fire_trunc, _info = env.step(1)
            if fire_reward != 0 or fire_done or fire_trunc:
                raise RuntimeError('Startup FIRE unexpectedly altered score or ended game')
        prev = preprocess(obs)
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    trace: list[dict] = []
    capped = False
    action_right_count = 0
    entropy_sum = 0.0
    rally_states: list[np.ndarray] = []
    rally_actions: list[int] = []
    rally_rewards: list[float] = []
    policy.eval()
    with torch.no_grad():
        while True:
            if cfg.game == 'cartpole':
                state = np.asarray(obs, dtype=np.float32)
                xx = torch.from_numpy(state).unsqueeze(0).to(device)
            else:
                cur = preprocess(obs)
                # Difference of two observations reveals movement direction.
                if cfg.pong_observation == 'stack':
                    # Both frames retain absolute paddle/ball position.
                    state = np.stack((prev, cur)).astype(np.int8)
                elif cfg.pong_observation == 'stackmotion':
                    # Absolute position plus explicit signed motion channel.
                    # The policy still learns from frames; no reward information
                    # or hand-coded action selection enters the observation.
                    state = np.stack((prev, cur, cur-prev)).astype(np.int8)
                else:
                    state = (cur - prev).astype(np.int8)
                xx = torch.from_numpy(state.astype(np.float32)).to(device)
                xx = xx.unsqueeze(0) if cfg.pong_observation != 'difference' else xx.view(1, 1, 80, 80)
            logits = policy(xx)
            dist = Categorical(logits=logits)
            a = int(logits.argmax(dim=-1).item()) if deterministic else int(dist.sample().item())
            action_right_count += int(a == 0)
            entropy_sum += float(dist.entropy().item())
            env_action = a if cfg.game == 'cartpole' else (2 if a == 0 else 3)
            next_obs, reward, terminated, truncated, _info = env.step(env_action)
            states.append(state)
            actions.append(a)
            rewards.append(float(reward))
            if len(trace) < trace_limit:
                trace.append({
                    'step': len(rewards),
                    'state': state.round(4).tolist() if cfg.game == 'cartpole'
                             else ({('two_frame_nonzero_pixels' if cfg.pong_observation == 'stack' else 'frame_channel_nonzero_pixels'): [int(np.count_nonzero(x)) for x in state]}
                                   if cfg.pong_observation in ('stack', 'stackmotion')
                                   else {'difference_nonzero_pixels': int(np.count_nonzero(state))}),
                    'policy_probabilities': torch.softmax(logits[0], dim=-1).cpu().tolist(),
                    'network_action': a, 'env_action': env_action,
                    'reward': float(reward), 'done': bool(terminated or truncated)})
            if on_point is not None:
                rally_states.append(state)
                rally_actions.append(a)
                rally_rewards.append(float(reward))
                if reward != 0:
                    # Score marks the end of a Pong point. All actions in this
                    # rally were sampled from the current on-policy model.
                    # Update NOW instead of accumulating entire games.
                    on_point({'states': np.stack(rally_states),
                              'actions': np.asarray(rally_actions, dtype=np.int64),
                              'rewards': rally_rewards[:]})
                    rally_states.clear()
                    rally_actions.clear()
                    rally_rewards.clear()
                    policy.eval()
            if cfg.game == 'pong':
                prev = cur
            obs = next_obs
            if terminated or truncated:
                break
            if cfg.max_steps > 0 and len(rewards) >= cfg.max_steps:
                capped = True
                break
    # Store images as signed bytes (-1/0/1); convert to floats only for training.
    return {'states': np.stack(states), 'actions': np.asarray(actions, dtype=np.int64),
            'rewards': rewards, 'score': float(sum(rewards)),
            'length': len(rewards), 'capped': capped, 'trace': trace,
            'action_right_fraction': action_right_count / max(1, len(rewards)),
            'mean_policy_entropy': entropy_sum / max(1, len(rewards))}


def update_from_batch(model, opt, batch, cfg, device, running_baseline):
    """Compute exact gradient of loss over saved actions via chunked backward.

    The *reward-to-go* policy-gradient estimator is used throughout. We do not
    use a critic, replay buffer, PPO, or an external pretrained agent.
    """
    states = np.concatenate([r['states'] for r in batch], axis=0)
    actions = np.concatenate([r['actions'] for r in batch], axis=0)
    targets = np.concatenate([returns_for_episode(r['rewards'], cfg) for r in batch])
    # Constant baseline comes from PREVIOUS batches so it cannot depend on
    # any action from the present batch. This keeps the score estimator unbiased.
    if cfg.baseline == 'running' and running_baseline is not None:
        advantages = targets - running_baseline
    else:
        advantages = targets.copy()
    # IMPORTANT: do not use within-batch mean as baseline here; including a
    # sample's own return in its baseline can introduce bias, particularly for
    # tiny batch sizes. No per-episode reward standardization is used.
    model.train()
    opt.zero_grad(set_to_none=True)
    total = len(actions)
    for start in range(0, total, cfg.train_chunk):
        end = min(start+cfg.train_chunk, total)
        x = torch.from_numpy(states[start:end].astype(np.float32))
        if cfg.game == 'pong' and cfg.pong_observation == 'difference':
            x = x.unsqueeze(1)
        x = x.to(device)
        y = torch.from_numpy(actions[start:end]).to(device)
        weight = torch.from_numpy(advantages[start:end]).to(device)
        logits = model(x)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(y)
        # Optional exploration bonus.  This is zero by default to preserve
        # the original assignment's pure REINFORCE objective.
        loss = (-(log_probs * weight).sum()
                - cfg.entropy_coef * dist.entropy().sum()) / total
        # OPTIONAL auxiliary visual guidance. This is NOT pure REINFORCE:
        # it derives directional labels from frame geometry, not rewards.
        # Do not use it for an unmodified assignment-objective run.
        if cfg.game == 'pong' and cfg.pong_guide_coef > 0:
            geo, usable = visual_geometry(x)
            vertical_error = geo[:, 6]
            valid = (usable > 0) & (vertical_error.abs() > 0.06)
            if bool(valid.any()):
                # Positive vertical_error means ball is BELOW right paddle.
                # action2_is_down is calibrated from game observations.
                down_index = 0 if cfg.action2_is_down else 1
                up_index = 1 - down_index
                labels = torch.where(vertical_error > 0,
                                     torch.full_like(y, down_index),
                                     torch.full_like(y, up_index))
                guide = nn.functional.cross_entropy(logits[valid], labels[valid], reduction='sum')
                loss = loss + cfg.pong_guide_coef * guide / total
        loss.backward()
    grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=10).item())
    opt.step()
    # EMA of returns-to-go of prior batches, used as constant baseline.
    batch_mean = float(np.mean(targets))
    if running_baseline is None:
        updated_baseline = batch_mean
    else:
        updated_baseline = 0.9 * running_baseline + 0.1 * batch_mean
    return updated_baseline, grad_norm


def make_plots(training: list[dict], evaluation: list[float], out: Path, cfg: Config):
    out.mkdir(parents=True, exist_ok=True)
    episode_scores = np.array([x['reward'] for x in training])
    rolling = np.array([episode_scores[max(0,i-99):i+1].mean() for i in range(len(episode_scores))])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(np.arange(1,len(episode_scores)+1), episode_scores, alpha=0.35,
            linewidth=0.8, label='Episode reward')
    ax.plot(np.arange(1,len(episode_scores)+1), rolling, linewidth=2,
            label='Moving average (up to 100 episodes)')
    ax.set(xlabel='Training episode', ylabel='Undiscounted total episode reward',
           title=f'{cfg.game}: REINFORCE ({cfg.baseline} baseline)')
    ax.grid(alpha=0.2); ax.legend(); fig.tight_layout()
    fig.savefig(out / 'training_curve.png', dpi=190); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.7))
    if cfg.game == 'cartpole':
        ax.hist(evaluation, bins=np.linspace(0, 500, 26), edgecolor='white')
    else:
        ax.hist(evaluation, bins=np.arange(-21.5,22.5,1), edgecolor='white')
    ax.set(xlabel='Undiscounted episode reward', ylabel='Number of evaluation episodes',
           title=f'{cfg.game}: {len(evaluation)}-episode final evaluation ({cfg.baseline})')
    ax.grid(axis='y', alpha=0.2); fig.tight_layout()
    fig.savefig(out / 'evaluation_histogram.png', dpi=190); plt.close(fig)


def write_csv(rows, out):
    if not rows:
        return
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)


def evaluate(env, model, cfg: Config, device, count, trace_first=False):
    records = []
    trace = []
    for ep in range(count):
        rr = rollout(env, model, cfg, episode_seed=cfg.seed+100000+ep,
                     device=device, deterministic=not cfg.eval_stochastic,
                     trace_limit=10 if (ep == 0 and trace_first) else 0)
        records.append({'episode': ep+1, 'reward': rr['score'],
                        'length': rr['length'], 'artificial_cap': int(rr['capped']),
                        'action_right_fraction': rr['action_right_fraction'],
                        'mean_policy_entropy': rr['mean_policy_entropy']})
        if ep == 0:
            trace = rr['trace']
        if (ep+1) % 100 == 0:
            print(f'  Evaluation completed: {ep+1}/{count}', flush=True)
    return records, trace


def train(cfg: Config):
    if cfg.pong_network not in ('classic','geometry'):
        raise ValueError('Invalid Pong network')
    if cfg.pong_network == 'geometry' and cfg.game == 'pong' and cfg.pong_observation == 'difference':
        raise ValueError('Geometry network requires stack or stackmotion observations')
    if cfg.pong_guide_coef < 0:
        raise ValueError('Guide coefficient must be nonnegative')
    if cfg.pong_guide_coef and (cfg.game != 'pong' or cfg.pong_network != 'geometry'):
        raise ValueError('Auxiliary guidance requires Pong geometry network')
    if cfg.pong_return_mode not in ('episode', 'point'):
        raise ValueError('pong_return_mode must be episode or point')
    if cfg.pong_observation not in ('difference', 'stack', 'stackmotion'):
        raise ValueError('pong_observation must be difference, stack, or stackmotion')
    if cfg.pong_update_mode not in ('episode', 'rally'):
        raise ValueError('pong_update_mode must be episode or rally')
    if cfg.pong_update_mode == 'rally' and (cfg.game != 'pong' or cfg.pong_return_mode != 'point'):
        raise ValueError('rally updates require Pong and --pong-return-mode point')
    if cfg.entropy_coef < 0:
        raise ValueError('Entropy coefficient must be nonnegative')
    if cfg.rallies_per_update < 1:
        raise ValueError('rallies_per_update must be positive')
    if cfg.batch_episodes < 1 or cfg.train_chunk < 1:
        raise ValueError('Batch size and chunk size must be positive')
    if cfg.eval_episodes < 1:
        raise ValueError('Evaluation episode count must be positive')
    if cfg.episodes < 1:
        raise ValueError('Training episode count must be positive')
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = choose_device(cfg.device)
    print(f'Environment: {cfg.game}, baseline: {cfg.baseline}, device: {device}', flush=True)
    env = build_env(cfg.game)
    if cfg.pong_guide_coef:
        cfg.action2_is_down = calibrate_action2_direction(env, cfg.seed, cfg.start_fire)
        print('Calibrated: Pong action 2 moves paddle ' + ('DOWN' if cfg.action2_is_down else 'UP'), flush=True)
    model = make_policy(cfg.game, cfg.pong_observation, cfg.pong_network).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    baseline = None
    training = []
    grad_norms = []
    first_trace = []
    batch = []
    start_ep = 0
    policy_updates = 0
    pending_rallies = []
    if cfg.resume_from is not None:
        ckpt = torch.load(cfg.resume_from, map_location='cpu', weights_only=False)
        older = ckpt['config']
        for key in ('game','baseline','gamma','seed','lr','batch_episodes',
                    'pong_observation','entropy_coef','pong_return_mode', 'pong_update_mode', 'rallies_per_update', 'pong_network', 'pong_guide_coef', 'action2_is_down'):
            if older.get(key, None) != getattr(cfg, key):
                raise ValueError(f'Resume configuration mismatch: {key}')
        model.load_state_dict(ckpt['state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_ep = int(ckpt['completed_episodes'])
        baseline = ckpt['baseline']
        training = ckpt['training']
        grad_norms = ckpt.get('gradient_norms', [])
        policy_updates = ckpt.get('policy_updates', len(grad_norms))
        first_trace = ckpt.get('first_trace', [])
        if 'torch_rng_state' in ckpt:
            torch.set_rng_state(ckpt['torch_rng_state'])
        if device.type == 'mps' and ckpt.get('mps_rng_state') is not None:
            torch.mps.set_rng_state(ckpt['mps_rng_state'])
        print(f'Resumed at episode {start_ep}; target episodes {cfg.episodes}', flush=True)
        if cfg.episodes < start_ep:
            raise ValueError('Target episodes cannot be less than completed episodes')
    try:
        for ep in range(start_ep, cfg.episodes):
            def flush_rallies():
                nonlocal baseline, policy_updates
                if not pending_rallies:
                    return
                # rollout samples actions under no_grad; recompute log-probs
                # and enable autograd for a single batched REINFORCE update.
                with torch.enable_grad():
                    baseline, norm = update_from_batch(model, opt, pending_rallies, cfg, device, baseline)
                grad_norms.append(norm)
                policy_updates += 1
                pending_rallies.clear()

            def update_on_point(rally):
                pending_rallies.append(rally)
                if len(pending_rallies) >= cfg.rallies_per_update:
                    flush_rallies()

            rr = rollout(env, model, cfg, episode_seed=cfg.seed+ep, device=device,
                         deterministic=False, trace_limit=10 if ep==0 else 0,
                         on_point=update_on_point if cfg.pong_update_mode == 'rally' else None)
            if cfg.pong_update_mode == 'rally':
                # No completed rally is silently discarded at game boundary.
                flush_rallies()
            if ep == 0:
                first_trace = rr['trace']
            training.append({'episode': ep+1, 'reward': rr['score'],
                             'length': rr['length'], 'artificial_cap': int(rr['capped']),
                             'action_right_fraction': rr['action_right_fraction'],
                             'mean_policy_entropy': rr['mean_policy_entropy']})
            if cfg.pong_update_mode == 'episode':
                batch.append(rr)
            if cfg.pong_update_mode == 'episode' and (len(batch) == cfg.batch_episodes or ep == cfg.episodes-1):
                baseline, norm = update_from_batch(model, opt, batch, cfg, device, baseline)
                grad_norms.append(norm)
                policy_updates += 1
                batch.clear()
                if cfg.checkpoint_every > 0 and (ep + 1) % cfg.checkpoint_every == 0:
                    # Intermediate checkpoints are not a claim of convergence.
                    # Save current progress so runs interrupted early still
                    # retain weights and their observed reward curve.
                    torch.save({'state_dict': model.state_dict(),
                                'optimizer_state_dict': opt.state_dict(),
                                'config': asdict(cfg), 'completed_episodes': ep + 1,
                                'baseline': baseline, 'training': training,
                                'gradient_norms': grad_norms,
                                'first_trace': first_trace,
                                'torch_rng_state': torch.get_rng_state(),
                                'mps_rng_state': (torch.mps.get_rng_state() if device.type == 'mps' else None)
                                }, out / 'checkpoint_latest.pt')
                    write_csv(training, out / 'training_progress.csv')
            if cfg.pong_update_mode == 'rally' and cfg.checkpoint_every > 0 and (ep+1) % cfg.checkpoint_every == 0:
                torch.save({'state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'config': asdict(cfg), 'completed_episodes': ep + 1,
                            'baseline': baseline, 'training': training,
                            'gradient_norms': grad_norms,
                            'policy_updates': policy_updates,
                            'first_trace': first_trace,
                            'torch_rng_state': torch.get_rng_state(),
                            'mps_rng_state': (torch.mps.get_rng_state() if device.type == 'mps' else None)},
                           out / 'checkpoint_latest.pt')
                write_csv(training, out / 'training_progress.csv')
            interval = 50 if cfg.game == 'cartpole' else 10
            if (ep+1) % interval == 0 or ep == 0:
                mean100 = np.mean([r['reward'] for r in training[-100:]])
                print(f'Train {ep+1}/{cfg.episodes}: reward={rr["score"]:.1f} '
                      f'avg(last up to 100)={mean100:.2f} '
                      f'length={rr["length"]} '
                      f'action0_fraction={rr["action_right_fraction"]:.2f} '
                      f'entropy={rr["mean_policy_entropy"]:.3f} '
                      f'updates={policy_updates}', flush=True)
        torch.save({'state_dict': model.state_dict(), 'config': asdict(cfg)}, out / 'policy.pt')
        eval_rows, eval_trace = evaluate(env, model, cfg, device,
                                         cfg.eval_episodes, trace_first=True)
    finally:
        env.close()
    write_csv(training, out / 'training.csv')
    write_csv(eval_rows, out / 'evaluation.csv')
    make_plots(training, [r['reward'] for r in eval_rows], out, cfg)
    eval_scores = np.asarray([r['reward'] for r in eval_rows], dtype=float)
    train_scores = np.asarray([r['reward'] for r in training], dtype=float)
    std = float(eval_scores.std(ddof=1)) if len(eval_scores)>1 else 0.
    results = {
        'environment': 'CartPole-v1' if cfg.game=='cartpole' else 'ALE/Pong-v5',
        'game': cfg.game, 'baseline': cfg.baseline, 'config': asdict(cfg),
        'pytorch_version': torch.__version__, 'device_resolved': str(device),
        'train_last100_mean': float(train_scores[-100:].mean()),
        'train_first100_mean': float(train_scores[:100].mean()),
        'train_episode_count': len(training), 'train_capped_count': sum(r['artificial_cap'] for r in training),
        'eval_episode_count': len(eval_rows), 'eval_mean': float(eval_scores.mean()),
        'eval_std_sample': std, 'eval_standard_error': std / np.sqrt(len(eval_scores)),
        'eval_ci95_normal': [float(eval_scores.mean()-1.96*std/np.sqrt(len(eval_scores))),
                             float(eval_scores.mean()+1.96*std/np.sqrt(len(eval_scores)))],
        'eval_min': float(eval_scores.min()), 'eval_max': float(eval_scores.max()),
        'eval_capped_count': sum(r['artificial_cap'] for r in eval_rows),
        'gradient_norm_mean_before_clipping': float(np.mean(grad_norms)) if grad_norms else 0.,
        'policy_updates': policy_updates,
        'baseline_last_ema_return': float(baseline),
        'eval_mean_action_right_fraction': float(np.mean([r['action_right_fraction'] for r in eval_rows])),
        'eval_mean_policy_entropy': float(np.mean([r['mean_policy_entropy'] for r in eval_rows])),
        'pong_return_mode': cfg.pong_return_mode,
        'evaluation_method': ('stochastic sampling from two-action policy'
                              if cfg.eval_stochastic else 'deterministic argmax of two-action policy'),
        'trace_train_first10': first_trace, 'trace_eval_first10': eval_trace,
        'pong_startup_fire_not_learned': cfg.start_fire if cfg.game=='pong' else False,
    }
    (out / 'metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({k:results[k] for k in ['game','baseline','eval_episode_count',
                          'eval_mean','eval_std_sample','eval_ci95_normal',
                          'train_capped_count','eval_capped_count']}, indent=2), flush=True)
    print(f'Plots and model saved in {out}', flush=True)


def evaluate_existing(cfg: Config):
    out = Path(cfg.out)
    checkpoint = torch.load(out/'policy.pt', map_location='cpu', weights_only=False)
    old_cfg = checkpoint['config']
    cfg.start_fire = old_cfg['start_fire']
    cfg.max_steps = old_cfg['max_steps']
    device = choose_device(cfg.device)
    cfg.pong_observation = old_cfg.get('pong_observation', 'difference')
    cfg.pong_network = old_cfg.get('pong_network', 'classic')
    model = make_policy(cfg.game, cfg.pong_observation, cfg.pong_network).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    env = build_env(cfg.game)
    try:
        records, _ = evaluate(env, model, cfg, device, cfg.eval_episodes, trace_first=True)
    finally:
        env.close()
    write_csv(records, out/'evaluation.csv')
    scores = np.asarray([r['reward'] for r in records],dtype=float)
    previous = json.loads((out/'metrics.json').read_text())
    previous['eval_episode_count'] = len(scores)
    previous['eval_mean'] = float(scores.mean())
    previous['eval_std_sample'] = float(scores.std(ddof=1)) if len(scores)>1 else 0.0
    previous['eval_standard_error'] = float(previous['eval_std_sample']/np.sqrt(len(scores)))
    previous['eval_ci95_normal'] = [previous['eval_mean']-1.96*previous['eval_standard_error'],
                                    previous['eval_mean']+1.96*previous['eval_standard_error']]
    previous['eval_min'], previous['eval_max'] = float(scores.min()), float(scores.max())
    previous['eval_capped_count'] = sum(r['artificial_cap'] for r in records)
    previous['eval_mean_action_right_fraction'] = float(np.mean([r['action_right_fraction'] for r in records]))
    previous['eval_mean_policy_entropy'] = float(np.mean([r['mean_policy_entropy'] for r in records]))
    previous['evaluation_method'] = ('stochastic sampling from two-action policy'
                                     if cfg.eval_stochastic else 'deterministic argmax of two-action policy')
    (out/'metrics.json').write_text(json.dumps(previous,indent=2))
    training = list(csv.DictReader((out/'training.csv').open()))
    training = [{'reward':float(x['reward'])} for x in training]
    make_plots(training, scores, out, cfg)
    print('Updated', out / 'metrics.json')


def diagnose_pong_input(cfg: Config):
    """Save a visual audit of real Atari frames entering the Pong policy."""
    if cfg.game != 'pong':
        raise ValueError('Input visualization is available for Pong only')
    env = build_env('pong')
    try:
        obs, _ = env.reset(seed=cfg.seed)
        if cfg.start_fire:
            obs, _, done, truncated, _ = env.step(1)
            if done or truncated:
                raise RuntimeError('Startup FIRE ended diagnostic episode')
        previous = preprocess(obs)
        panels = []
        for t in range(31):
            current = preprocess(obs)
            if t in (1, 10, 20, 30):
                panels.append((t, obs.copy(), current.copy(), (current-previous).copy()))
            previous = current
            # This is an observation check, NOT a performance evaluation.
            obs, _, done, truncated, _ = env.step(2 if (t//5)%2==0 else 3)
            if done or truncated:
                break
    finally:
        env.close()
    if not panels:
        raise RuntimeError('No Pong observations were produced')
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(len(panels), 3, figsize=(10, 3*len(panels)), squeeze=False)
    report = []
    for i, (t, raw, cur, diff) in enumerate(panels):
        axs[i,0].imshow(raw)
        axs[i,0].set_title(f'Original RGB: step {t}')
        axs[i,1].imshow(cur, cmap='gray', vmin=0, vmax=1)
        axs[i,1].set_title('Preprocessed 80×80')
        axs[i,2].imshow(diff, cmap='gray', vmin=-1, vmax=1)
        axs[i,2].set_title('Signed motion: current - previous')
        for ax in axs[i]:
            ax.set_axis_off()
        report.append({'step': t,
                       'foreground_pixels': int(np.count_nonzero(cur)),
                       'motion_pixels': int(np.count_nonzero(diff))})
    fig.tight_layout()
    target = out/'pong_input_audit.png'
    fig.savefig(target, dpi=150)
    plt.close(fig)
    (out/'pong_input_audit.json').write_text(json.dumps(report, indent=2))
    print(f'Pong observations saved to: {target}', flush=True)
    print(json.dumps(report, indent=2), flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['train','evaluate','diagnose'])
    p.add_argument('--game', choices=['cartpole','pong'], required=True)
    p.add_argument('--baseline', choices=['none','running'], default='none')
    p.add_argument('--episodes', type=int, default=None,
                   help='Training episodes (CartPole 1500, Pong 2000 by default)')
    p.add_argument('--batch-episodes', type=int, default=None)
    p.add_argument('--lr', type=float, default=None,
                   help='Defaults: CartPole 0.01, Pong 0.0003 (stability adjustment)')
    p.add_argument('--eval-episodes', type=int, default=500)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='auto', choices=['auto','cpu','mps','cuda'])
    p.add_argument('--train-chunk', type=int, default=256)
    p.add_argument('--max-steps', type=int, default=0,
                   help='0 means no artificial truncation (needed for true full games)')
    p.add_argument('--no-start-fire', action='store_true',
                   help='Disable fixed startup FIRE in Pong; learned actions are still 2 and 3')
    p.add_argument('--pong-observation', choices=['difference','stack','stackmotion'], default='difference',
                   help='stack retains the previous and current binary Pong frames (2 channels)')
    p.add_argument('--pong-network', choices=['classic', 'geometry'], default='classic',
                   help='geometry network explicitly retains tiny ball/paddle coordinates from frames')
    p.add_argument('--pong-guide-coef', type=float, default=0.0,
                   help='EXPERIMENTAL: visual paddle-tracking auxiliary loss (not pure REINFORCE)')
    p.add_argument('--rallies-per-update' , type=int, default=5,
                   help='Number of scored Pong rallies per policy update in rally mode')
    p.add_argument('--pong-update-mode', choices=['episode', 'rally'], default='episode',
                   help='Experimental: update REINFORCE immediately after each Pong point')
    p.add_argument('--pong-return-mode', choices=['episode','point'], default='episode',
                   help='Pong credit assignment: full episode (assignment objective) or rally/point reset (variant)')
    p.add_argument('--entropy-coef', type=float, default=0.,
                   help='Exploration bonus; 0 preserves the original objective')
    p.add_argument('--eval-stochastic', action='store_true',
                   help='Sample policy actions during evaluation, as in training')
    p.add_argument('--checkpoint-every', type=int, default=0,
                   help='Save latest intermediate weights and training log every N episodes')
    p.add_argument('--resume-from', default=None,
                   help='Continue from checkpoint_latest.pt, preserving previous training history')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    cfg = Config(game=a.game, baseline=a.baseline,
                 episodes=a.episodes if a.episodes is not None else (1500 if a.game=='cartpole' else 2000),
                 batch_episodes=a.batch_episodes if a.batch_episodes is not None else (10 if a.game=='cartpole' else 2),
                 lr=a.lr if a.lr is not None else (0.01 if a.game=='cartpole' else 0.0003),
                 gamma=0.95 if a.game=='cartpole' else 0.99,
                 seed=a.seed, device=a.device, eval_episodes=a.eval_episodes,
                 train_chunk=a.train_chunk, max_steps=a.max_steps,
                 start_fire=not a.no_start_fire,
                 out=a.out or f'runs/{a.game}_{a.baseline}',
                 pong_observation=a.pong_observation,
                 entropy_coef=a.entropy_coef,
                 eval_stochastic=a.eval_stochastic,
                 checkpoint_every=a.checkpoint_every,
                 resume_from=a.resume_from,
                 pong_return_mode=a.pong_return_mode,
                 pong_update_mode=a.pong_update_mode,
                 rallies_per_update=a.rallies_per_update,
                 pong_network=a.pong_network, pong_guide_coef=a.pong_guide_coef)
    return a.command, cfg


if __name__ == '__main__':
    command, config = parse()
    if command == 'train':
        train(config)
    elif command == 'evaluate':
        evaluate_existing(config)
    else:
        diagnose_pong_input(config)
