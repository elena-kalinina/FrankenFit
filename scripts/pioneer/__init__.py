# Franken-Fit × Pioneer (Fastino Labs) training toolkit.
#
# Scripts in this package manage the Day-1 → Day-2 fine-tune loop:
#
#   train.py   — dataset prep + LoRA fine-tune + eval + side-by-side infer
#   probe.py   — quick side-by-side inference against any two models
#
# Run from the repo root with the venv active:
#   python -m scripts.pioneer.train --help
#   python -m scripts.pioneer.probe --help
