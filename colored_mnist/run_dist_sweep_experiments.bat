@echo off
REM run_dist_sweep_experiments.bat
REM Final experiments for Experiment 7 — run after search completes
REM Edit hyperparameters below with values from search_results_dist_sweep_*.json

echo ============================================================
echo Experiment 7 -- Distribution Distance Sweep -- Experiments
echo ============================================================

REM --- Compute distances first (only needs to run once) ---
echo.
echo [Computing distribution distances]
python compute_distances.py --n_samples 1000

REM ---------------------------------------------------------------
REM IRM Oracle — fill in best hparams from search for each e value
REM ---------------------------------------------------------------

REM Example for e=0.1 — replace hparams with search results
python main_dist_sweep.py --mode irm --e_train 0.1 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.2 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.3 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.4 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.5 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.6 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.7 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.8 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.9 --selection_method oracle --n_restarts 10

REM --- IRM Train Domain Val ---
python main_dist_sweep.py --mode irm --e_train 0.1 --selection_method train_domain_val --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.2 --selection_method train_domain_val --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.3 --selection_method train_domain_val --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.4 --selection_method train_domain_val --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.5 --selection_method train_domain_val --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.6 --selection_method train_domain_val --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.7 --selection_method train_domain_val --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.8 --selection_method train_domain_val --n_restarts 10
python main_dist_sweep.py --mode irm --e_train 0.9 --selection_method train_domain_val --n_restarts 10

REM --- ERM Oracle (same hparams for all e values) ---
python main_dist_sweep.py --mode erm --e_train 0.1 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode erm --e_train 0.2 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode erm --e_train 0.3 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode erm --e_train 0.4 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode erm --e_train 0.5 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode erm --e_train 0.6 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode erm --e_train 0.7 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode erm --e_train 0.8 --selection_method oracle --n_restarts 10
python main_dist_sweep.py --mode erm --e_train 0.9 --selection_method oracle --n_restarts 10

REM --- Generate plots ---
echo.
echo [Generating plots]
python plot_dist_sweep.py --selection_method oracle
python plot_dist_sweep.py --selection_method train_domain_val

echo.
echo ============================================================
echo Experiments complete. Check experiment7_dist_sweep_*.png
echo ============================================================