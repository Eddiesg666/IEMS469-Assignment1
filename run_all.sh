#!/usr/bin/env bash
set -euo pipefail
# Real training + 500-episode evaluation for all 4 required experiments.
python shuttle.py --out . --mc-episodes 1000
python sensitivity.py
python -m pytest -q tests/
python validate_envs.py --game cartpole --episodes 20
python validate_envs.py --game pong --episodes 5
python pg.py train --game cartpole --baseline none --episodes 1500 --eval-episodes 500
python pg.py train --game cartpole --baseline running --episodes 1500 --eval-episodes 500
python pg.py train --game pong --baseline none --episodes 2000 --eval-episodes 500 --lr 0.0003
python pg.py train --game pong --baseline running --episodes 2000 --eval-episodes 500 --lr 0.0003
python compare.py
python build_report.py --username YOUR_GITHUB_USERNAME --repo YOUR_REPO_NAME
