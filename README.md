# IEMS 469 - Advanced Algorithms for Machine Learning - Assignment 1

Implemented in Python/PyTorch. Use the **actual** run results (not values copied into this README) when preparing the report. This repository includes the required shuttle MDP, policy-gradient experiments for CartPole-v1 and ALE/Pong-v5, an optional scalar running baseline, quantitative tests, and a clearly labeled **supplementary visually guided Pong experiment**.

## Setup and tests (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests/
python shuttle.py --out . --mc-episodes 1000
python sensitivity.py
python validate_envs.py --game cartpole --episodes 20
python validate_envs.py --game pong --episodes 5
```

The final revised test suite passed 30 tests on the user's Mac. Test suite files are kept under `tests/`. Some Matplotlib dependency deprecation warnings are benign.

## Required REINFORCE runs: CartPole-v1

Two separate from-scratch experiments, each with 1,500 training episodes, discount factor 0.95 and Adam learning rate 0.01. Deterministic greedy evaluation used for these CartPole runs.

```bash
python pg.py train --game cartpole --baseline none --episodes 1500 --eval-episodes 500
python pg.py train --game cartpole --baseline running --episodes 1500 --eval-episodes 500
```

## Required REINFORCE runs: ALE/Pong-v5

Two separate from-scratch experiments. Each uses 200 training episodes and a 500-game stochastic evaluation, discount factor 0.99, Adam learning rate 0.0003, 2 complete games per optimizer update, full-episode reward-to-go, geometry-aware policy on preprocessed consecutive 80x80 frames and signed frame differences, and **no** auxiliary visual guidance or entropy bonus. The policy chooses only actions 2 and 3 (action 1 FIRE may be used as a fixed startup step outside the learned policy).

```bash
python pg.py train \
  --game pong --baseline none --episodes 200 --batch-episodes 2 \
  --pong-observation stackmotion --pong-network geometry \
  --pong-guide-coef 0 --pong-update-mode episode --pong-return-mode episode \
  --entropy-coef 0 --lr 0.0003 --eval-stochastic \
  --eval-episodes 500 --checkpoint-every 50 --out runs/pong_none

python pg.py train \
  --game pong --baseline running --episodes 200 --batch-episodes 2 \
  --pong-observation stackmotion --pong-network geometry \
  --pong-guide-coef 0 --pong-update-mode episode --pong-return-mode episode \
  --entropy-coef 0 --lr 0.0003 --eval-stochastic \
  --eval-episodes 500 --checkpoint-every 50 --out runs/pong_running
```

A training run may be resumed with `--resume-from runs/<run>/checkpoint_latest.pt` and `--episodes` equal to the TOTAL desired episode count. Do not treat `--episodes` as additional episodes when resuming.

## Supplementary experiment: visually guided Pong

This is **not** the unchanged basic REINFORCE objective and must not replace either required Pong condition. The visual-guidance auxiliary loss derives direction targets from visible ball and paddle positions. It was combined with rally-level updates, point-level returns, an entropy bonus, and a geometry-aware CNN. The final model trained for 400 episodes and was independently evaluated over 500 games.

Key settings: `--pong-network geometry --pong-observation stackmotion --pong-guide-coef 0.5 --pong-update-mode rally --pong-return-mode point --rallies-per-update 5 --entropy-coef 0.001 --lr 0.0003 --baseline running`. The recorded training and evaluation results are under `runs/pong_v5_guided_probe/`. This extension provides **additional supervision**, so the observed improvement cannot be ascribed to reward-only policy gradients.

To evaluate the saved policy again without retraining:

```bash
python pg.py evaluate --game pong --baseline running --eval-stochastic \
  --eval-episodes 500 --out runs/pong_v5_guided_probe
```

## Results from completed runs

All entries below were evaluated for 500 episodes. Numbers are mean +/- sample standard deviation of **undiscounted episode rewards**.

| Experiment | Training episodes | Evaluation mean +/- sample SD |
|---|---:|---:|
| CartPole, no baseline | 1500 | 126.624 +/- 5.694 |
| CartPole, running baseline | 1500 | 500.000 +/- 0.000 |
| Pong, no baseline (reward only) | 200 | -20.560 +/- 0.647 |
| Pong, running baseline (reward only) | 200 | -20.156 +/- 0.924 |
| Pong, visually guided extension | 400 | -14.702 +/- 2.843 |

Interpretation: CartPole benefits strongly from the running baseline for this single training seed. Reward-only Pong remains close to random performance after 200 episodes. Visual guidance improves Pong in this particular experiment but modifies several factors at once; it is an enhanced method, not an isolated baseline ablation.

## Compare and generate the PDF report

```bash
python compare.py
python build_report.py --username YOUR_GITHUB_USERNAME --repo YOUR_REPO_NAME
```

The patched report generator checks for all four **required** 500-episode evaluation JSON files and the guided 500-episode result. It generates `assignment1_report.pdf` in the project root using real metrics and training/evaluation figures, describing the hybrid guided result separately. Review the generated PDF before submission. Add the actual GitHub repository link to the Canvas submission as required.

Expected key files for each run: `metrics.json`, `training.csv`, `evaluation.csv`, `training_curve.png`, `evaluation_histogram.png`, `policy.pt`, plus intermediate checkpoints if enabled. Do not delete `runs/` when updating code.

## Validation

- Shuttle MDP: finite-horizon recursion, value iteration and policy iteration agree numerically; stochastic Monte Carlo cost validates the Bellman result; threshold sensitivity tests vary dispatch/holding costs.
- CartPole and Pong: test preprocessing, return recursion, action restrictions, gradient direction, baseline behavior, and training/evaluation API paths. `validate_envs.py` prints the first ten steps.
- Additional Pong checks: inspect `runs/pong_input_audit/pong_input_audit.png` for visible paddles/ball and motion; check policy entropy, action fractions, checkpointed updates, and empirical rewards.
- Limitations: one training seed per condition; paired evaluation differences do not measure uncertainty across training seeds. The guided method changes the supervision and update schedule, so further controlled ablations are necessary.

All dependencies come from `requirements.txt` and are installable with ordinary pip packages. Source code and experiments should remain individually developed and documented according to course rules.
