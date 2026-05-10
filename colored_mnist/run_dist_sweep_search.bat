@echo off
REM run_dist_sweep_search.bat
REM Hyperparameter search for Experiment 7 — all (mode, e_train, selection) combinations
REM IRM: 50 trials per combination | ERM: fixed hyperparameters (1 trial)

echo ============================================================
echo Experiment 7 -- Distribution Distance Sweep -- HP Search
echo ============================================================

REM --- IRM Oracle ---
for %%e in (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) do (
    echo.
    echo [IRM / oracle / e=%%e]
    python search_dist_sweep.py --mode irm --e_train %%e --selection_method oracle --n_trials 50
)

REM --- IRM Train Domain Val ---
for %%e in (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) do (
    echo.
    echo [IRM / train_domain_val / e=%%e]
    python search_dist_sweep.py --mode irm --e_train %%e --selection_method train_domain_val --n_trials 50
)

REM --- ERM (fixed hparams, only needs one pass) ---
for %%e in (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) do (
    echo.
    echo [ERM / oracle / e=%%e]
    python search_dist_sweep.py --mode erm --e_train %%e --selection_method oracle --n_trials 1
)

echo.
echo ============================================================
echo Search complete. Check search_results_dist_sweep_*.json
echo ============================================================