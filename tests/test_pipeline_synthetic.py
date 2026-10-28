"""End-to-end code-path checks with toy environments; NOT a Pong score claim."""
from pathlib import Path
import json
import numpy as np
import pytest
from pg import Config, train


class ToySpace:
    def seed(self, s):self.seed_value=s


class ToyGame:
    def __init__(self, game):
        self.game=game
        self.action_space=ToySpace()
        self.num=0
    def reset(self,seed=None):
        self.num=0
        return self.obs(),{}
    def obs(self):
        if self.game=='cartpole':
            return np.array([.1*self.num, -.03,.04,0.05],dtype=np.float32)
        x=np.zeros((210,160,3),dtype=np.uint8)
        x[50+self.num,20:25,0]=255
        return x
    def step(self,action):
        if self.game=='pong' and action==1 and self.num==0:
            return self.obs(),0.,False,False,{}
        assert action in ((0,1) if self.game=='cartpole' else (2,3))
        self.num+=1
        return self.obs(),float(action in (1,2)),self.num>=5,False,{}
    def close(self):pass


@pytest.mark.parametrize('game',['cartpole','pong'])
@pytest.mark.parametrize('baseline',['none','running'])
def test_complete_pipeline_with_tiny_toy_env(monkeypatch,tmp_path,game,baseline):
    import pg
    monkeypatch.setattr(pg,'build_env',lambda g:ToyGame(g))
    out=tmp_path/'output'
    cfg=Config(game=game,baseline=baseline,episodes=3,batch_episodes=1,
               lr=.001,gamma=.95 if game=='cartpole' else .99,seed=19,
               device='cpu',eval_episodes=3,train_chunk=4,max_steps=0,
               start_fire=True,out=str(out))
    train(cfg)
    for name in ('policy.pt','metrics.json','training.csv','evaluation.csv',
                 'training_curve.png','evaluation_histogram.png'):
        assert (out/name).exists(),name
    result=json.loads((out/'metrics.json').read_text())
    assert result['eval_episode_count']==3
    assert result['train_episode_count']==3
    assert result['eval_capped_count']==0
    assert len(result['trace_train_first10'])==5
    assert result['trace_train_first10'][0]['env_action'] in ((0,1) if game=='cartpole' else (2,3))
