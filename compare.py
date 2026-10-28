"""Summarize four experimental runs; generate baseline comparison plots."""
from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('runs')
OUT=Path('figures')
OUT.mkdir(parents=True,exist_ok=True)
summary=[]
for game in ('cartpole','pong'):
    metrics={}; series={}; evaluations={}
    for baseline in ('none','running'):
        folder=BASE/f'{game}_{baseline}'
        if not (folder/'metrics.json').exists():
            print('Not available yet:',folder/'metrics.json')
            continue
        metrics[baseline]=json.loads((folder/'metrics.json').read_text())
        with (folder/'training.csv').open() as f:
            series[baseline]=np.array([float(x['reward']) for x in csv.DictReader(f)])
        with (folder/'evaluation.csv').open() as f:
            evaluations[baseline]=np.array([float(x['reward']) for x in csv.DictReader(f)])
        m=metrics[baseline]
        summary.append({'game':game,'baseline':baseline,'n':m['eval_episode_count'],
                        'mean':m['eval_mean'],'sd':m['eval_std_sample'],
                        'train_first100':m['train_first100_mean'],
                        'train_last100':m['train_last100_mean']})
    if not metrics:continue
    fig,ax=plt.subplots(figsize=(9,4.6))
    for baseline, scores in series.items():
        rolling=np.array([scores[max(0,i-99):i+1].mean() for i in range(len(scores))])
        ax.plot(np.arange(1,len(scores)+1),rolling,label=f'{baseline} baseline (100-episode mean)')
    ax.set(title=f'{game}: learning curves with vs without running baseline',
           xlabel='Training episode',ylabel='100-episode moving average reward')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(OUT/f'{game}_baseline_comparison.png',dpi=190);plt.close(fig)
    if len(evaluations)==2:
        x,y=evaluations['none'],evaluations['running']
        if len(x)==len(y):
            diff=y-x
            mean=float(diff.mean()); se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
            print(f'{game}: paired differences [running minus none] '
                  f'{mean:.3f}, approximate 95% CI ({mean-1.96*se:.3f}, {mean+1.96*se:.3f})')
            summary.append({'game':game,'contrast':'running minus none','paired_mean':mean,
                            'paired_ci95':[mean-1.96*se,mean+1.96*se]})
print(json.dumps(summary,indent=2))
(Path('results')/'comparison.json').write_text(json.dumps(summary,indent=2))
