"""Fast Pong diagnostics for the Advanced Algorithms for ML assignment.

Put this file next to your CURRENT pg.py. Run:
    python pong_sanity_check.py
This does not train the real Pong agent or modify any runs/ files.
"""
from pathlib import Path
import numpy as np
import torch
import pg


def synthetic_policy_gradient_test():
    """Show whether the exact PongPolicy + update_from_batch learns a visual two-action task."""
    torch.set_num_threads(2)
    pg.set_seed(2026)
    model = pg.make_policy('pong', 'stackmotion').cpu()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    cfg = pg.Config(game='pong', baseline='running', episodes=30, batch_episodes=32,
                    lr=.001, gamma=.99, seed=2026, device='cpu', eval_episodes=1,
                    train_chunk=64, max_steps=0, start_fire=False, out='unused',
                    pong_observation='stackmotion', pong_return_mode='point')
    images = np.zeros((2, 3, 80, 80), dtype=np.int8)
    for label, x in enumerate([20, 60]):
        previous = np.zeros((80, 80), dtype=np.int8)
        current = previous.copy()
        previous[30:33, x-2:x+1] = 1
        current[30:33, x:x+3] = 1
        previous[30:42, 70:72] = current[30:42, 70:72] = 1
        images[label] = np.stack([previous, current, current-previous])

    def probability_correct():
        with torch.no_grad():
            p = torch.softmax(model(torch.tensor(images, dtype=torch.float32)), -1).numpy()
        return float(p[0, 0]), float(p[1, 1])

    before = probability_correct()
    baseline = None
    for _ in range(30):
        batch = []
        for _ in range(32):
            label = int(np.random.randint(2))
            with torch.no_grad():
                prob = torch.softmax(model(torch.tensor(images[label:label+1], dtype=torch.float32)), -1)[0]
                action = int(torch.multinomial(prob, 1).item())
            r = 1.0 if action == label else -1.0
            batch.append({'states': images[label:label+1],
                          'actions': np.array([action], dtype=np.int64),
                          'rewards': [r]})
        baseline, _ = pg.update_from_batch(model, opt, batch, cfg, torch.device('cpu'), baseline)
    after = probability_correct()
    print('Visual REINFORCE test: probabilities of correct action')
    print('   Before training: [{:.3f}, {:.3f}]'.format(*before))
    print('   After 30 updates: [{:.3f}, {:.3f}]'.format(*after))
    print('   RESULT:', 'PASS' if min(after) > .90 else 'NEEDS INVESTIGATION')


def paddle_y(frame):
    """Estimate center of the right-side Pong paddle from the binary screen."""
    scores = frame[:, 65:78].sum(axis=0)
    if scores.size == 0 or scores.max() < 4:
        return None
    col = int(scores.argmax()) + 65
    rows = np.flatnonzero(frame[:, col] != 0)
    return round(float(np.mean(rows)), 2) if rows.size else None


def action_effect_test():
    """Compare both Atari actions on identical seeded game starts."""
    for action in (2, 3):
        env = pg.build_env('pong')
        try:
            obs, _ = env.reset(seed=2026)
            obs, _, _, _, _ = env.step(1)
            centers = []
            for i in range(40):
                obs, reward, done, trunc, _ = env.step(action)
                if i in (0, 5, 10, 20, 30, 39):
                    centers.append((i+1, paddle_y(pg.preprocess(obs))))
                if done or trunc:
                    break
            print('Environment action {}: right-side object y-coordinates {}'.format(action, centers))
        finally:
            env.close()
    print('If the sequences differ, the environment is responding differently to actions 2 and 3.')
    print('These estimates are approximate: inspect pong_input_audit.png if no paddle is detected.')


if __name__ == '__main__':
    print('Loaded pg.py from:', Path(pg.__file__).resolve())
    print('Has rally support:', hasattr(pg.Config, '__dataclass_fields__') and 'rallies_per_update' in pg.Config.__dataclass_fields__)
    synthetic_policy_gradient_test()
    print('\nChecking real Pong controls...')
    try:
        action_effect_test()
    except (ImportError, RuntimeError) as exc:
        print('Environment check unavailable here:', exc)
