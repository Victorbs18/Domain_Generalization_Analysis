@echo off
echo ============================================================
echo   Experiment 5 -- Proximity vs Diversity 2x2
echo   Run from colored_mnist/ root:
echo   src\exp5\run_exp5.bat > results\exp5\results.txt 2>&1
echo ============================================================

echo.
echo --- Hyperparameter search (IRM only) ---

echo.
echo [IRM - Config A - train_domain_val]
python src/exp5/search_exp5.py --mode irm --config A --selection_method train_domain_val --n_trials 50

echo.
echo [IRM - Config A - leave_one_domain_out]
python src/exp5/search_exp5.py --mode irm --config A --selection_method leave_one_domain_out --n_trials 50

echo.
echo [IRM - Config B - oracle]
python src/exp5/search_exp5.py --mode irm --config B --selection_method oracle --n_trials 50

echo.
echo [IRM - Config B - train_domain_val]
python src/exp5/search_exp5.py --mode irm --config B --selection_method train_domain_val --n_trials 50

echo.
echo [IRM - Config B - leave_one_domain_out]
python src/exp5/search_exp5.py --mode irm --config B --selection_method leave_one_domain_out --n_trials 50

echo.
echo [IRM - Config C - oracle]
python src/exp5/search_exp5.py --mode irm --config C --selection_method oracle --n_trials 50

echo.
echo [IRM - Config C - train_domain_val]
python src/exp5/search_exp5.py --mode irm --config C --selection_method train_domain_val --n_trials 50

echo.
echo [IRM - Config C - leave_one_domain_out]
python src/exp5/search_exp5.py --mode irm --config C --selection_method leave_one_domain_out --n_trials 50

echo.
echo [IRM - Config D - oracle]
python src/exp5/search_exp5.py --mode irm --config D --selection_method oracle --n_trials 50

echo.
echo [IRM - Config D - train_domain_val]
python src/exp5/search_exp5.py --mode irm --config D --selection_method train_domain_val --n_trials 50

echo.
echo [IRM - Config D - leave_one_domain_out]
python src/exp5/search_exp5.py --mode irm --config D --selection_method leave_one_domain_out --n_trials 50

echo.
echo --- Final experiments (IRM) ---

echo.
echo [IRM - Config A - oracle]
python src/exp5/main_exp5.py --mode irm --config A --selection_method oracle --n_restarts 10

echo.
echo [IRM - Config A - train_domain_val]
python src/exp5/main_exp5.py --mode irm --config A --selection_method train_domain_val --n_restarts 10

echo.
echo [IRM - Config A - leave_one_domain_out]
python src/exp5/main_exp5.py --mode irm --config A --selection_method leave_one_domain_out --n_restarts 10

echo.
echo [IRM - Config B - oracle]
python src/exp5/main_exp5.py --mode irm --config B --selection_method oracle --n_restarts 10

echo.
echo [IRM - Config B - train_domain_val]
python src/exp5/main_exp5.py --mode irm --config B --selection_method train_domain_val --n_restarts 10

echo.
echo [IRM - Config B - leave_one_domain_out]
python src/exp5/main_exp5.py --mode irm --config B --selection_method leave_one_domain_out --n_restarts 10

echo.
echo [IRM - Config C - oracle]
python src/exp5/main_exp5.py --mode irm --config C --selection_method oracle --n_restarts 10

echo.
echo [IRM - Config C - train_domain_val]
python src/exp5/main_exp5.py --mode irm --config C --selection_method train_domain_val --n_restarts 10

echo.
echo [IRM - Config C - leave_one_domain_out]
python src/exp5/main_exp5.py --mode irm --config C --selection_method leave_one_domain_out --n_restarts 10

echo.
echo [IRM - Config D - oracle]
python src/exp5/main_exp5.py --mode irm --config D --selection_method oracle --n_restarts 10

echo.
echo [IRM - Config D - train_domain_val]
python src/exp5/main_exp5.py --mode irm --config D --selection_method train_domain_val --n_restarts 10

echo.
echo [IRM - Config D - leave_one_domain_out]
python src/exp5/main_exp5.py --mode irm --config D --selection_method leave_one_domain_out --n_restarts 10

echo.
echo --- Final experiments (ERM) ---

echo.
python src/exp5/main_exp5.py --mode erm --config A --selection_method oracle --n_restarts 10
python src/exp5/main_exp5.py --mode erm --config B --selection_method oracle --n_restarts 10
python src/exp5/main_exp5.py --mode erm --config C --selection_method oracle --n_restarts 10
python src/exp5/main_exp5.py --mode erm --config D --selection_method oracle --n_restarts 10

echo.
echo All experiments finished.