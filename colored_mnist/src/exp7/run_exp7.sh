#!/bin/bash
# run_exp7.sh — Experiment 7: Two Correlated Spurious Features
# Run from colored_mnist/ root:
#   bash src/exp7/run_exp7.sh > results/exp7/results.txt 2>&1

echo "============================================================"
echo "  Experiment 7 -- Two Correlated Spurious Features"
echo "  Color + Texture both correlated with label at rate e"
echo "  4 configs: A={0.1,0.2} B={0.7,0.8} C={0.1,0.5} D={0.1,0.8}"
echo "============================================================"

echo ""
echo "--- Hyperparameter search ---"

for CONFIG in A B C D; do
    echo "[IRM - Config ${CONFIG} - oracle]"
    python3 src/exp7/search_exp7.py --config ${CONFIG} --mode irm --selection_method oracle --n_trials 50

    echo "[IRM - Config ${CONFIG} - train_domain_val]"
    python3 src/exp7/search_exp7.py --config ${CONFIG} --mode irm --selection_method train_domain_val --n_trials 50

    echo "[IRM - Config ${CONFIG} - leave_one_domain_out]"
    python3 src/exp7/search_exp7.py --config ${CONFIG} --mode irm --selection_method leave_one_domain_out --n_trials 50

    echo "[ERM - Config ${CONFIG} - oracle]"
    python3 src/exp7/search_exp7.py --config ${CONFIG} --mode erm --selection_method oracle
done

echo ""
echo "--- Final experiments ---"

for CONFIG in A B C D; do
    echo "[IRM - Config ${CONFIG} - oracle]"
    python3 src/exp7/main_exp7.py --config ${CONFIG} --mode irm --selection_method oracle --n_restarts 10

    echo "[IRM - Config ${CONFIG} - train_domain_val]"
    python3 src/exp7/main_exp7.py --config ${CONFIG} --mode irm --selection_method train_domain_val --n_restarts 10

    echo "[IRM - Config ${CONFIG} - leave_one_domain_out]"
    python3 src/exp7/main_exp7.py --config ${CONFIG} --mode irm --selection_method leave_one_domain_out --n_restarts 10

    echo "[ERM - Config ${CONFIG} - oracle]"
    python3 src/exp7/main_exp7.py --config ${CONFIG} --mode erm --selection_method oracle --n_restarts 10

    echo "[ERM - Config ${CONFIG} - train_domain_val]"
    python3 src/exp7/main_exp7.py --config ${CONFIG} --mode erm --selection_method train_domain_val --n_restarts 10

    echo "[ERM - Config ${CONFIG} - leave_one_domain_out]"
    python3 src/exp7/main_exp7.py --config ${CONFIG} --mode erm --selection_method leave_one_domain_out --n_restarts 10
done

echo ""
echo "All experiments finished."
