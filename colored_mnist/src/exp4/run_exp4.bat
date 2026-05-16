@echo off
echo ============================================================
echo   Experiment 4 -- Number of Environments (Fixed Range)
echo   Run from colored_mnist/ root:
echo   src\exp4\run_exp4.bat > results\exp4\results.txt 2>&1
echo ============================================================

echo.
echo --- Hyperparameter search ---

echo.
echo [IRM - n_envs=1 - oracle]
python src/exp4/search_exp4.py --mode irm --n_envs 1 --selection_method oracle --n_trials 50

echo.
echo [IRM - n_envs=1 - train_domain_val]
python src/exp4/search_exp4.py --mode irm --n_envs 1 --selection_method train_domain_val --n_trials 50

echo.
echo [IRM - n_envs=1 - leave_one_domain_out]
python src/exp4/search_exp4.py --mode irm --n_envs 1 --selection_method leave_one_domain_out --n_trials 50

echo.
echo [IRM - n_envs=2 - train_domain_val]
python src/exp4/search_exp4.py --mode irm --n_envs 2 --selection_method train_domain_val --n_trials 50

echo.
echo [IRM - n_envs=2 - leave_one_domain_out]
python src/exp4/search_exp4.py --mode irm --n_envs 2 --selection_method leave_one_domain_out --n_trials 50

echo.
echo [IRM - n_envs=3 - oracle]
python src/exp4/search_exp4.py --mode irm --n_envs 3 --selection_method oracle --n_trials 50

echo.
echo [IRM - n_envs=3 - train_domain_val]
python src/exp4/search_exp4.py --mode irm --n_envs 3 --selection_method train_domain_val --n_trials 50

echo.
echo [IRM - n_envs=3 - leave_one_domain_out]
python src/exp4/search_exp4.py --mode irm --n_envs 3 --selection_method leave_one_domain_out --n_trials 50

echo.
echo [IRM - n_envs=4 - oracle]
python src/exp4/search_exp4.py --mode irm --n_envs 4 --selection_method oracle --n_trials 50

echo.
echo [IRM - n_envs=4 - train_domain_val]
python src/exp4/search_exp4.py --mode irm --n_envs 4 --selection_method train_domain_val --n_trials 50

echo.
echo [IRM - n_envs=4 - leave_one_domain_out]
python src/exp4/search_exp4.py --mode irm --n_envs 4 --selection_method leave_one_domain_out --n_trials 50

echo.
echo [IRM - n_envs=5 - oracle]
python src/exp4/search_exp4.py --mode irm --n_envs 5 --selection_method oracle --n_trials 50

echo.
echo [IRM - n_envs=5 - train_domain_val]
python src/exp4/search_exp4.py --mode irm --n_envs 5 --selection_method train_domain_val --n_trials 50

echo.
echo [IRM - n_envs=5 - leave_one_domain_out]
python src/exp4/search_exp4.py --mode irm --n_envs 5 --selection_method leave_one_domain_out --n_trials 50

echo.
echo --- Final experiments ---

echo.
echo [IRM - n_envs=1 - oracle]
python src/exp4/main_exp4.py --mode irm --n_envs 1 --selection_method oracle --n_restarts 10

echo.
echo [IRM - n_envs=1 - train_domain_val]
python src/exp4/main_exp4.py --mode irm --n_envs 1 --selection_method train_domain_val --n_restarts 10

echo.
echo [IRM - n_envs=1 - leave_one_domain_out]
python src/exp4/main_exp4.py --mode irm --n_envs 1 --selection_method leave_one_domain_out --n_restarts 10

echo.
echo [IRM - n_envs=2 - oracle]
python src/exp4/main_exp4.py --mode irm --n_envs 2 --selection_method oracle --n_restarts 10

echo.
echo [IRM - n_envs=2 - train_domain_val]
python src/exp4/main_exp4.py --mode irm --n_envs 2 --selection_method train_domain_val --n_restarts 10

echo.
echo [IRM - n_envs=2 - leave_one_domain_out]
python src/exp4/main_exp4.py --mode irm --n_envs 2 --selection_method leave_one_domain_out --n_restarts 10

echo.
echo [IRM - n_envs=3 - oracle]
python src/exp4/main_exp4.py --mode irm --n_envs 3 --selection_method oracle --n_restarts 10

echo.
echo [IRM - n_envs=3 - train_domain_val]
python src/exp4/main_exp4.py --mode irm --n_envs 3 --selection_method train_domain_val --n_restarts 10

echo.
echo [IRM - n_envs=3 - leave_one_domain_out]
python src/exp4/main_exp4.py --mode irm --n_envs 3 --selection_method leave_one_domain_out --n_restarts 10

echo.
echo [IRM - n_envs=4 - oracle]
python src/exp4/main_exp4.py --mode irm --n_envs 4 --selection_method oracle --n_restarts 10

echo.
echo [IRM - n_envs=4 - train_domain_val]
python src/exp4/main_exp4.py --mode irm --n_envs 4 --selection_method train_domain_val --n_restarts 10

echo.
echo [IRM - n_envs=4 - leave_one_domain_out]
python src/exp4/main_exp4.py --mode irm --n_envs 4 --selection_method leave_one_domain_out --n_restarts 10

echo.
echo [IRM - n_envs=5 - oracle]
python src/exp4/main_exp4.py --mode irm --n_envs 5 --selection_method oracle --n_restarts 10

echo.
echo [IRM - n_envs=5 - train_domain_val]
python src/exp4/main_exp4.py --mode irm --n_envs 5 --selection_method train_domain_val --n_restarts 10

echo.
echo [IRM - n_envs=5 - leave_one_domain_out]
python src/exp4/main_exp4.py --mode irm --n_envs 5 --selection_method leave_one_domain_out --n_restarts 10

echo.
echo [ERM - all n_envs - oracle]
python src/exp4/main_exp4.py --mode erm --n_envs 1 --selection_method oracle --n_restarts 10
python src/exp4/main_exp4.py --mode erm --n_envs 2 --selection_method oracle --n_restarts 10
python src/exp4/main_exp4.py --mode erm --n_envs 3 --selection_method oracle --n_restarts 10
python src/exp4/main_exp4.py --mode erm --n_envs 4 --selection_method oracle --n_restarts 10
python src/exp4/main_exp4.py --mode erm --n_envs 5 --selection_method oracle --n_restarts 10

echo.
echo All experiments finished.