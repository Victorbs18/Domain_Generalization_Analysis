@echo off

echo ============================================================
echo   Two Correlated Spurious Features — Hyperparameter Search
echo ============================================================
echo   Spurious feature 1: color (e controls correlation)
echo   Spurious feature 2: background texture (same e as color)
echo   Causal feature: digit shape
echo ============================================================

echo --- Config: original e=[0.1, 0.2] ---
python search_texture.py --config original --mode erm --selection_method oracle
python search_texture.py --config original --mode irm --selection_method oracle --n_trials 50
python search_texture.py --config original --mode irm --selection_method train_domain_val --n_trials 50
python search_texture.py --config original --mode irm --selection_method leave_one_domain_out --n_trials 50

echo --- Config: diverse e=[0.1, 0.5] ---
python search_texture.py --config diverse --mode erm --selection_method oracle
python search_texture.py --config diverse --mode irm --selection_method oracle --n_trials 50
python search_texture.py --config diverse --mode irm --selection_method train_domain_val --n_trials 50
python search_texture.py --config diverse --mode irm --selection_method leave_one_domain_out --n_trials 50

echo --- Config: proximate e=[0.7, 0.8] ---
python search_texture.py --config proximate --mode erm --selection_method oracle
python search_texture.py --config proximate --mode irm --selection_method oracle --n_trials 50
python search_texture.py --config proximate --mode irm --selection_method train_domain_val --n_trials 50
python search_texture.py --config proximate --mode irm --selection_method leave_one_domain_out --n_trials 50

echo All texture searches finished.