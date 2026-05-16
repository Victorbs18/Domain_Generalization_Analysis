@echo off
echo ============================================================
echo   Experiment 1 — IRM Reproduction
echo   Run from colored_mnist/ root:
echo   run_exp1.bat ^> results/exp1/results.txt 2^>^&1
echo ============================================================

echo.
echo --- Hyperparameter search ---
echo.
echo [IRM]
python src/exp1/search_exp1.py --mode irm --n_trials 50

echo.
echo [ERM]
python src/exp1/search_exp1.py --mode erm

echo.
echo [Grayscale]
python src/exp1/search_exp1.py --mode grayscale --n_trials 50

echo.
echo --- Final experiments ---
echo.
echo [IRM]
python src/exp1/main_exp1.py --mode irm --n_restarts 10

echo.
echo [ERM]
python src/exp1/main_exp1.py --mode erm --n_restarts 10

echo.
echo [Grayscale]
python src/exp1/main_exp1.py --mode grayscale --n_restarts 10

echo.
echo All experiments finished.