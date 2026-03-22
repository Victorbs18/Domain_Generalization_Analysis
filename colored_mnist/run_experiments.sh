#!/bin/bash
# run_experiments.sh
#
# Runs all three Colored-MNIST conditions from the IRM paper.
#
# Workflow:
#   1. python search.py --mode irm       --n_trials 50
#   2. python search.py --mode grayscale --n_trials 50
#   3. python search.py --mode erm
#   4. Replace the hyperparameters below with your search results
#   5. bash run_experiments.sh

#!/bin/bash

set -e

echo "============================================================"
echo "  IRM"
echo "============================================================"
python -u main.py \
  --hidden_dim=256 \
  --l2_regularizer_weight=0.0002589933531580329 \
  --lr=0.0011620802131636504 \
  --penalty_anneal_iters=65 \
  --penalty_weight=34583.84531216486 \
  --steps=401 \
  --n_restarts=10

echo ""
echo "============================================================"
echo "  ERM (baseline)"
echo "============================================================"
python -u main.py \
  --hidden_dim=256 \
  --l2_regularizer_weight=0.001 \
  --lr=0.001 \
  --penalty_anneal_iters=0 \
  --penalty_weight=0.0 \
  --steps=501 \
  --n_restarts=10

echo ""
echo "============================================================"
echo "  ERM grayscale (oracle)"
echo "============================================================"
python -u main.py \
  --grayscale_model \
  --hidden_dim=255 \
  --l2_regularizer_weight=0.001304656360450818 \
  --lr=0.0019086620171489411 \
  --penalty_anneal_iters=0 \
  --penalty_weight=0.0 \
  --steps=401 \
  --n_restarts=10

echo ""
echo "All experiments finished."