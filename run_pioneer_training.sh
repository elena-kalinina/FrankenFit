#!/usr/bin/env bash
# Quick-start: kick off the Pioneer LoRA fine-tune from existing seed data.
# Run from the repo root with the venv active:
#   source .venv/bin/activate
#   bash run_pioneer_training.sh
#
# Skips Pioneer's /generate step (which can queue for 10–18 min) and goes
# straight to training on the 90+ seed rows already in func_test/out/.
# After training, set PIONEER_TRAINED_MODEL_ID=<job_id> in .env.

set -euo pipefail

python -m scripts.pioneer.train \
  --phase=train \
  --dataset-name=frankenfit-binary-v3 \
  --model-name=frankenfit-binary-lora \
  --skip-generate \
  "$@"
