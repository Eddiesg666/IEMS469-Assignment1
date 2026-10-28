"""Controlled perturbations to the shuttle problem validate policy behavior."""
from pathlib import Path
import json
import numpy as np
from shuttle import make_mdp,value_iteration


def threshold(policy):
    chosen=np.flatnonzero(policy)
    return int(chosen[0]) if chosen.size else None

cases=[
    ('default',100,2),
    ('less_expensive_shuttle',50,2),
    ('more_expensive_shuttle',200,2),
    ('higher_waiting_cost',100,5),
    ('lower_waiting_cost',100,1),
    ('free_shuttle',0,2),
]
output=[]
for tag,cf,ch in cases:
    p,c=make_mdp(dispatch_cost=cf,holding_cost=ch)
    v,pi,iterations,delta=value_iteration(p,c)
    output.append({'case':tag,'dispatch_cost':cf,'holding_cost':ch,
                   'threshold':threshold(pi),'value_at_40':float(v[40]),
                   'iterations':iterations})
by={r['case']:r for r in output}
# Lower shuttle price/higher holding price should not delay dispatch.
assert by['less_expensive_shuttle']['threshold'] <= by['default']['threshold']
assert by['higher_waiting_cost']['threshold'] <= by['default']['threshold']
assert by['more_expensive_shuttle']['threshold'] >= by['default']['threshold']
assert by['lower_waiting_cost']['threshold'] >= by['default']['threshold']
assert by['free_shuttle']['threshold']==1
Path('results').mkdir(exist_ok=True)
Path('results/shuttle_sensitivity.json').write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))
