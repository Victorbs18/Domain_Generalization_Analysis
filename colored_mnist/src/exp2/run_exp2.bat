@echo off
echo ============================================================
echo   Experiment 2 -- Model Selection Methods
echo   Run from colored_mnist/ root:
echo   src\exp2\run_exp2.bat > results\exp2\results.txt 2>&1
echo ============================================================

echo.
echo --- Hyperparameter search ---
echo.
echo [IRM - train_domain_val]
python src/exp2/search_exp2.py --mode irm --selection_method train_domain_val --n_trials 50

echo.
echo [IRM - leave_one_domain_out]
python src/exp2/search_exp2.py --mode irm --selection_method leave_one_domain_out --n_trials 50

echo.
echo [ERM - train_domain_val]
python src/exp2/search_exp2.py --mode erm --selection_method train_domain_val

echo.
echo [ERM - leave_one_domain_out]
python src/exp2/search_exp2.py --mode erm --selection_method leave_one_domain_out

echo.
echo [Grayscale - train_domain_val]
python src/exp2/search_exp2.py --mode grayscale --selection_method train_domain_val --n_trials 50

echo.
echo [Grayscale - leave_one_domain_out]
python src/exp2/search_exp2.py --mode grayscale --selection_method leave_one_domain_out --n_trials 50

echo.
echo --- Final experiments ---
echo.
echo [IRM - oracle]
python src/exp2/main_exp2.py --mode irm --selection_method oracle --n_restarts 10

echo.
echo [IRM - train_domain_val]
python src/exp2/main_exp2.py --mode irm --selection_method train_domain_val --n_restarts 10

echo.
echo [IRM - leave_one_domain_out]
python src/exp2/main_exp2.py --mode irm --selection_method leave_one_domain_out --n_restarts 10

echo.
echo [ERM - oracle]
python src/exp2/main_exp2.py --mode erm --selection_method oracle --n_restarts 10

echo.
echo [ERM - train_domain_val]
python src/exp2/main_exp2.py --mode erm --selection_method train_domain_val --n_restarts 10

echo.
echo [ERM - leave_one_domain_out]
python src/exp2/main_exp2.py --mode erm --selection_method leave_one_domain_out --n_restarts 10

echo.
echo [Grayscale - oracle]
python src/exp2/main_exp2.py --mode grayscale --selection_method oracle --n_restarts 10

echo.
echo [Grayscale - train_domain_val]
python src/exp2/main_exp2.py --mode grayscale --selection_method train_domain_val --n_restarts 10

echo.
echo [Grayscale - leave_one_domain_out]
python src/exp2/main_exp2.py --mode grayscale --selection_method leave_one_domain_out --n_restarts 10

echo.
echo All experiments finished.