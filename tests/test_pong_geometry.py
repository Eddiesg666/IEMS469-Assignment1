"""New v5 tests; keep every earlier test in the user's tests/ folder."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from torch import nn
import pg


def frame(ball=(40, 30), paddle_y=40, opp_y=40):
    x = torch.zeros(3, 80, 80, dtype=torch.float32)
    for t in [0, 1]:
        x[t, paddle_y-4:paddle_y+5, 73:75] = 1.0
        x[t, opp_y-4:opp_y+5, 7:9] = 1.0
    x[0, ball[1]-1, ball[0]-1] = 1.0
    x[1, ball[1], ball[0]] = 1.0
    x[2] = x[1] - x[0]
    return x


def test_position_extraction_tracks_single_ball_and_paddle():
    a = frame(ball=(20, 22), paddle_y=50).unsqueeze(0)
    b = frame(ball=(45, 60), paddle_y=50).unsqueeze(0)
    fa, valid_a = pg.visual_geometry(a)
    fb, valid_b = pg.visual_geometry(b)
    assert valid_a.item() == valid_b.item() == 1
    assert fa[0, 3].item() < fb[0, 3].item()  # ball_y
    assert fa[0, 2].item() < fb[0, 2].item()  # ball_x
    assert fa[0, 6].item() < 0 < fb[0, 6].item()  # ball/paddle relative y
    assert abs(fa[0, 4].item()) > 0   # x velocity
    assert abs(fa[0, 5].item()) > 0   # y velocity


def test_geometry_absent_ball_is_safe():
    x = frame()
    x[0:2, :, 12:69] = 0
    x[2] = x[1]-x[0]
    ft, good = pg.visual_geometry(x.unsqueeze(0))
    assert good.item() == 0
    assert torch.isfinite(ft).all()


def test_pong_geometry_policy_backpropagates():
    model = pg.PongGeometryPolicy(3)
    x = torch.stack([frame((30, 15), 50), frame((55, 65), 50)])
    out = model(x)
    assert out.shape == (2, 2)
    loss = nn.functional.cross_entropy(out, torch.tensor([0, 1]))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_pong_geometry_learns_ball_height_with_supervision():
    # Isolated visual-learning test, *not* evidence of reinforcement success.
    torch.manual_seed(11)
    model = pg.PongGeometryPolicy(3)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    x = torch.stack([frame((20, 16), 50), frame((35, 64), 50)])
    labels = torch.tensor([0, 1])
    with torch.no_grad():
        initial = torch.softmax(model(x), dim=-1)
    for _ in range(45):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), labels)
        loss.backward(); opt.step()
    with torch.no_grad():
        final = torch.softmax(model(x), dim=-1)
    assert final[0, 0] > 0.95 and final[1, 1] > 0.95
    assert float(final[0, 0]) > float(initial[0, 0])
    assert float(final[1, 1]) > float(initial[1, 1])


def test_auxiliary_guide_changes_gradient_without_changing_base_rewards(tmp_path):
    common=dict(game='pong', baseline='none', episodes=1,batch_episodes=1,
                lr=0.0001, gamma=0.99, seed=7, device='cpu',
                eval_episodes=1, train_chunk=16, max_steps=0,start_fire=False,
                out=str(tmp_path),pong_observation='stackmotion',
                pong_network='geometry',pong_guide_coef=0.5,action2_is_down=True)
    cfg=pg.Config(**common)
    images=np.stack([frame((25,20),50).numpy(), frame((30,65),50).numpy()])
    batch=[{'states':images,'actions':np.array([0,1]),'rewards':[0.0,-1.0]}]
    model=pg.make_policy('pong','stackmotion','geometry')
    opt=torch.optim.Adam(model.parameters(), lr=0.001)
    base, grad=pg.update_from_batch(model,opt,batch,cfg,torch.device('cpu'),None)
    assert np.isfinite(base) and grad>0


def test_differences_require_stacked_frames():
    with pytest.raises(ValueError):
        pg.PongGeometryPolicy(1)
