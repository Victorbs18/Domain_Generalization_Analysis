#!/bin/bash
# run_exp6.sh — Experiment 6: Performance vs Distribution Distance
# Run from colored_mnist/ root:
#   bash src/exp6/run_exp6.sh > results/exp6/results.txt 2>&1

echo "============================================================"
echo "  Experiment 6 -- Performance vs Distribution Distance"
echo "  e1=0.1 fixed, e2 swept [0.2..0.9], test e=0.9"
echo "============================================================"

echo ""
echo "--- Step 1: Compute distances ---"
python3 src/exp6/compute_distances.py

echo ""
echo "--- Step 2: Hyperparameter search ---"

for E2 in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo "[IRM - e2=${E2} - oracle]"
    python3 src/exp6/search_exp6.py --mode irm --e2 ${E2} --selection_method oracle --n_trials 50

    echo "[IRM - e2=${E2} - train_domain_val]"
    python3 src/exp6/search_exp6.py --mode irm --e2 ${E2} --selection_method train_domain_val --n_trials 50

    echo "[IRM - e2=${E2} - leave_one_domain_out]"
    python3 src/exp6/search_exp6.py --mode irm --e2 ${E2} --selection_method leave_one_domain_out --n_trials 50
done

echo ""
echo "--- Step 3: Final experiments ---"

for E2 in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo "[IRM - e2=${E2} - oracle]"
    python3 src/exp6/main_exp6.py --mode irm --e2 ${E2} --selection_method oracle --n_restarts 10

    echo "[IRM - e2=${E2} - train_domain_val]"
    python3 src/exp6/main_exp6.py --mode irm --e2 ${E2} --selection_method train_domain_val --n_restarts 10

    echo "[IRM - e2=${E2} - leave_one_domain_out]"
    python3 src/exp6/main_exp6.py --mode irm --e2 ${E2} --selection_method leave_one_domain_out --n_restarts 10

    echo "[ERM - e2=${E2} - oracle]"
    python3 src/exp6/main_exp6.py --mode erm --e2 ${E2} --selection_method oracle --n_restarts 10

    echo "[ERM - e2=${E2} - train_domain_val]"
    python3 src/exp6/main_exp6.py --mode erm --e2 ${E2} --selection_method train_domain_val --n_restarts 10

    echo "[ERM - e2=${E2} - leave_one_domain_out]"
    python3 src/exp6/main_exp6.py --mode erm --e2 ${E2} --selection_method leave_one_domain_out --n_restarts 10
done

echo ""
echo "--- Step 4: Plot ---"
python3 src/exp6/plot_exp6.py

echo ""
echo "All experiments finished."
