"""Run environment and experimental sanity checks; requires Gymnasium + ALE."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pg import Config, build_env, make_policy, rollout, choose_device


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--game', choices=['cartpole','pong'], required=True)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--out',default='results')
    a=p.parse_args()
    cfg=Config(game=a.game,baseline='none',episodes=0,batch_episodes=1,
               lr=.01,gamma=.95 if a.game=='cartpole' else .99,seed=2026,
               device='cpu',eval_episodes=0,train_chunk=128,max_steps=0,
               start_fire=True,out='')
    env=build_env(a.game)
    model=make_policy(a.game)
    scores=[]; lengths=[]; traces=[]
    try:
        for i in range(a.episodes):
            rr=rollout(env,model,cfg,episode_seed=2026+100000+i,
                       device=torch.device('cpu'),deterministic=False,
                       trace_limit=10 if i==0 else 0)
            scores.append(rr['score']); lengths.append(rr['length'])
            assert not rr['capped']
            if i==0: traces=rr['trace']
    finally:
        env.close()
    print(f'{a.game} randomly initialized policy: {a.episodes} episodes, '
          f'mean reward={np.mean(scores):.3f}, sample std={np.std(scores,ddof=1) if len(scores)>1 else 0:.3f}')
    print('First 10 environment steps:')
    for x in traces: print(json.dumps(x))
    result={'game':a.game, 'experiment':'UNTRAINED policy control',
            'episodes':a.episodes,'mean_reward':float(np.mean(scores)),
            'sample_std':float(np.std(scores,ddof=1)) if len(scores)>1 else 0,
            'episode_rewards':scores,'episode_lengths':lengths,
            'first10_trace':traces}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/f'{a.game}_random_control.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
