"""Synthetic-only checks of revised Pong integration; not a real Atari score."""
import json
import numpy as np
import torch
from pg import Config, PongPolicy, train
from test_pipeline_synthetic import ToyGame


def test_two_frame_pong_policy_finite_gradients():
    model=PongPolicy(input_channels=2)
    inputs=torch.randn(2,2,80,80)
    logits=model(inputs)
    assert logits.shape==(2,2)
    loss=(-torch.log_softmax(logits,dim=1)[:,0]).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_stacked_pong_training_evaluation_and_intermediate_checkpoint(monkeypatch,tmp_path):
    import pg
    monkeypatch.setattr(pg,'build_env',lambda game: ToyGame(game))
    folder=tmp_path/'pong'
    cfg=Config(game='pong',baseline='running',episodes=4,batch_episodes=2,
               lr=.0001,gamma=.99,seed=7,device='cpu',eval_episodes=3,
               train_chunk=3,max_steps=0,start_fire=True,out=str(folder),
               pong_observation='stack',entropy_coef=.01,
               eval_stochastic=True,checkpoint_every=2)
    train(cfg)
    d=json.loads((folder/'metrics.json').read_text())
    assert d['config']['pong_observation']=='stack'
    assert d['config']['entropy_coef']==.01
    assert d['eval_episode_count']==3
    assert 'stochastic' in d['evaluation_method']
    assert 0<=d['trace_train_first10'][0]['policy_probabilities'][0]<=1
    assert 'two_frame_nonzero_pixels' in d['trace_train_first10'][0]['state']
    assert (folder/'checkpoint_latest.pt').is_file()
    assert (folder/'training_progress.csv').is_file()


def test_resume_uses_saved_training_history(monkeypatch,tmp_path):
    import pg
    monkeypatch.setattr(pg,'build_env',lambda game: ToyGame(game))
    folder=tmp_path/'pong_resumable'
    common=dict(game='pong',baseline='running',batch_episodes=2,lr=.0001,
                gamma=.99,seed=7,device='cpu',eval_episodes=2,
                train_chunk=3,max_steps=0,start_fire=True,out=str(folder),
                pong_observation='stack',entropy_coef=.01,
                eval_stochastic=True,checkpoint_every=2)
    train(Config(episodes=4,**common))
    assert (folder/'checkpoint_latest.pt').exists()
    train(Config(episodes=6,resume_from=str(folder/'checkpoint_latest.pt'),**common))
    metrics=json.loads((folder/'metrics.json').read_text())
    assert metrics['train_episode_count']==6
    assert metrics['train_last100_mean']==np.mean([float(r['reward']) for r in __import__('csv').DictReader(open(folder/'training.csv'))])


def test_evaluation_respects_stochastic_flag(monkeypatch, tmp_path):
    """Regression test: stochastic evaluation must not silently take argmax."""
    import pg
    seen = []
    def fake_rollout(env, policy, cfg, episode_seed, device, deterministic, trace_limit=0):
        seen.append(deterministic)
        return dict(score=0., length=2, capped=False,
                    action_right_fraction=0.5, mean_policy_entropy=0.69,
                    trace=[{'step':1}] if trace_limit else [])
    monkeypatch.setattr(pg, 'rollout', fake_rollout)
    defaults = dict(game='pong',baseline='none',episodes=1,batch_episodes=1,
                    lr=.0001,gamma=.99,seed=7,device='cpu',eval_episodes=3,
                    train_chunk=3,max_steps=0,start_fire=True,out=str(tmp_path))
    cfg=Config(eval_stochastic=True, **defaults)
    pg.evaluate(None,None,cfg,torch.device('cpu'),3)
    assert seen == [False, False, False]
    seen.clear()
    cfg.eval_stochastic=False
    pg.evaluate(None,None,cfg,torch.device('cpu'),3)
    assert seen == [True, True, True]
