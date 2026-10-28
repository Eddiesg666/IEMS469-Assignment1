"""Assignment 1 / Part 2: finite-horizon enumeration, VI, and PI.

All costs are *costs*, not rewards. Smaller V is better.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_mdp(capacity=15, dispatch_cost=100.0, holding_cost=2.0,
             max_people=200, arrivals=(1, 2, 3, 4, 5)):
    """Return costs C[201,2] and transition probabilities P[201,2,201]."""
    n = max_people + 1
    p = np.zeros((n, 2, n), dtype=float)
    c = np.zeros((n, 2), dtype=float)
    if len(arrivals) == 0:
        raise ValueError('At least one arrival value is required')
    for state in range(n):
        for action in (0, 1):
            remaining = max(0, state - capacity * action)
            c[state, action] = dispatch_cost * action + holding_cost * remaining
            for arrival in arrivals:
                next_state = min(max_people, remaining + int(arrival))
                p[state, action, next_state] += 1.0 / len(arrivals)
    assert np.allclose(p.sum(axis=2), 1.0), 'Rows should be probability distributions'
    return p, c


def q_values(p, c, v, gamma):
    return c + gamma * np.einsum('sak,k->sa', p, v)


def finite_horizon(p, c, gamma=0.95, horizon=500):
    """Enumerate each action and each arrival via backward induction."""
    v = np.zeros(p.shape[0])   # V_T(s) = 0
    first_policy = None
    for t in range(horizon - 1, -1, -1):
        q = q_values(p, c, v, gamma)
        first_policy = np.argmin(q, axis=1)
        v = np.min(q, axis=1)
    return v, first_policy


def value_iteration(p, c, gamma=0.95, tol=1e-11, max_iterations=100000):
    v = np.zeros(p.shape[0])
    for iteration in range(1, max_iterations + 1):
        new_v = np.min(q_values(p, c, v, gamma), axis=1)
        delta = float(np.max(np.abs(new_v - v)))
        v = new_v
        # Bound on ||V-V*|| for gamma contraction: gamma*delta/(1-gamma)
        if gamma * delta / (1.0 - gamma) < tol:
            return v, np.argmin(q_values(p, c, v, gamma), axis=1), iteration, delta
    raise RuntimeError('Value iteration did not meet requested convergence tolerance')


def policy_iteration(p, c, gamma=0.95, max_iterations=1000):
    n = p.shape[0]
    states = np.arange(n)
    policy = np.zeros(n, dtype=np.int64)  # initially always wait
    identity = np.eye(n)
    for iteration in range(1, max_iterations + 1):
        # Exact policy evaluation, not approximate iterative evaluation.
        v = np.linalg.solve(identity - gamma * p[states, policy], c[states, policy])
        improved = np.argmin(q_values(p, c, v, gamma), axis=1)
        if np.array_equal(improved, policy):
            return v, policy, iteration
        policy = improved
    raise RuntimeError('Policy iteration did not converge')


def simulate_cost(p, c, policy, initial_state=40, gamma=0.95, horizon=500,
                  episodes=1500, seed=123):
    """Monte Carlo estimate of the policy's discounted cost from one state."""
    rng = np.random.default_rng(seed)
    costs = np.empty(episodes)
    for i in range(episodes):
        s, total, discount = initial_state, 0.0, 1.0
        for _ in range(horizon):
            a = int(policy[s])
            total += discount * c[s, a]
            s = int(rng.choice(p.shape[2], p=p[s, a]))
            discount *= gamma
        costs[i] = total
    se = float(costs.std(ddof=1) / np.sqrt(episodes))
    return {'mean': float(costs.mean()), 'sample_std': float(costs.std(ddof=1)),
            'standard_error': se, 'ci95_lower': float(costs.mean()-1.96*se),
            'ci95_upper': float(costs.mean()+1.96*se), 'episodes': episodes}


def make_figures(v_finite, v_vi, policy_pi, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    states = np.arange(len(v_vi))
    plt.figure(figsize=(8.2, 4.7))
    plt.plot(states, v_finite, lw=2)
    plt.xlabel('Number of waiting passengers')
    plt.ylabel('Optimal expected discounted cost')
    plt.title('Finite horizon: enumeration / backward induction, T=500')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / 'shuttle_finite_value.png', dpi=190)
    plt.close()

    plt.figure(figsize=(8.2, 4.7))
    plt.plot(states, v_vi, lw=2)
    plt.xlabel('Number of waiting passengers')
    plt.ylabel('Optimal expected discounted cost')
    plt.title('Infinite horizon: value iteration')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / 'shuttle_infinite_value.png', dpi=190)
    plt.close()

    plt.figure(figsize=(8.2, 4.7))
    plt.step(states, policy_pi, where='mid', lw=1.8)
    plt.yticks([0, 1], ['Wait (0)', 'Dispatch (1)'])
    plt.ylim(-0.12, 1.12)
    plt.xlabel('Number of waiting passengers')
    plt.title('Infinite horizon: optimal policy from policy iteration')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / 'shuttle_policy.png', dpi=190)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='.')
    parser.add_argument('--mc-episodes', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    p, c = make_mdp()
    gamma = 0.95
    v_finite, pol_finite = finite_horizon(p, c, gamma=gamma, horizon=500)
    v_vi, pol_vi, vi_iters, final_change = value_iteration(p, c, gamma=gamma)
    v_pi, pol_pi, pi_iters = policy_iteration(p, c, gamma=gamma)
    residual = float(np.max(np.abs(v_vi - np.min(q_values(p, c, v_vi, gamma), axis=1))))
    finite_gap = float(np.max(np.abs(v_finite-v_vi)))
    pi_gap = float(np.max(np.abs(v_pi - v_vi)))
    greedy_policy_agreement = bool(np.array_equal(pol_vi, pol_pi))
    # Reward-based model checking: compare finite and infinite, PI/VI and controlled edge cases.
    assert residual < 1e-8
    assert finite_gap < 1e-5
    assert pi_gap < 1e-7
    assert greedy_policy_agreement
    assert pol_pi[0] == 0
    assert np.all(np.diff(v_vi) >= -1e-7), 'Higher queue size should not cost less'

    p_zero, c_zero = make_mdp(arrivals=(0,))
    v_zero, pol_zero, _, _ = value_iteration(p_zero, c_zero, gamma)
    assert pol_zero[0] == 0 and abs(v_zero[0]) < 1e-8
    # Gamma=0: only the immediate cost matters (checked against simple hand formula).
    v_myopic = np.min(c, axis=1)
    v0, _, _, _ = value_iteration(p, c, gamma=0)
    assert np.allclose(v0, v_myopic)
    # Zero dispatch cost: sending a shuttle weakly improves immediate cost and future queues.
    p_free, c_free = make_mdp(dispatch_cost=0)
    _, free_policy, _, _ = value_iteration(p_free, c_free, gamma)
    assert np.all(free_policy[1:] == 1)

    mc = simulate_cost(p, c, pol_pi, initial_state=40, gamma=gamma,
                       episodes=args.mc_episodes, seed=args.seed)
    always_wait = np.zeros(201, dtype=int)
    always_send = np.ones(201, dtype=int)
    mc_wait = simulate_cost(p, c, always_wait, episodes=args.mc_episodes,
                            initial_state=40, seed=args.seed)
    mc_send = simulate_cost(p, c, always_send, episodes=args.mc_episodes,
                            initial_state=40, seed=args.seed)
    optimal_v40 = float(v_pi[40])
    if args.mc_episodes >= 100:
        assert abs(mc['mean']-optimal_v40) <= 5.0 * mc['standard_error'] + 1.0, 'MC/model disagreement'
    make_figures(v_finite, v_vi, pol_pi, out / 'figures')
    dispatch_states = np.flatnonzero(pol_pi == 1)
    results = {'parameters': {'K': 15, 'cf': 100, 'ch': 2, 'gamma': gamma,
                               'max_waiting': 200, 'arrivals': [1,2,3,4,5], 'T': 500},
               'value_iteration_iterations': vi_iters, 'value_iteration_last_delta': final_change,
               'policy_iteration_iterations': pi_iters,
               'bellman_residual': residual,
               'finite_vs_infinite_max_abs_difference': finite_gap,
               'policy_iteration_vs_value_iteration_max_abs_difference': pi_gap,
               'policy_agreement': greedy_policy_agreement,
               'dispatch_states_min': int(dispatch_states.min()) if len(dispatch_states) else None,
               'dispatch_states_max': int(dispatch_states.max()) if len(dispatch_states) else None,
               'dispatch_states_count': int(len(dispatch_states)),
               'policy_is_single_threshold': bool(len(dispatch_states) and np.array_equal(
                    dispatch_states, np.arange(dispatch_states.min(), 201))),
               'value_at_state_0': float(v_vi[0]),
               'value_at_state_15': float(v_vi[15]),
               'value_at_state_40': optimal_v40,
               'value_at_state_200': float(v_vi[200]),
               'model_vs_monte_carlo_at_40': mc, 'always_wait_monte_carlo_at_40': mc_wait,
               'always_dispatch_monte_carlo_at_40': mc_send,
               'edge_tests': {'zero_queue_waits': True, 'deterministic_zero_arrivals_zero_value_at_0': True,
                              'myopic_gamma_zero': True, 'free_dispatch_all_positive_states': True,
                              'value_nondecreasing': True, 'transitions_valid': True}}
    with (out / 'results' / 'shuttle.json').open('w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved 3 plots under {out / 'figures'}")


if __name__ == '__main__':
    main()
