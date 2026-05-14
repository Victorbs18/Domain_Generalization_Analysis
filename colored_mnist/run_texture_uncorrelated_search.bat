@echo off

echo ============================================================
echo   Two INDEPENDENT Spurious Features — Hyperparameter Search
echo ============================================================
echo   e1: color weakly correlated, texture strongly correlated
echo   e2: color strongly correlated, texture weakly correlated
echo   Shape is the ONLY stable feature across environments
echo ============================================================

echo --- Config: original e=[0.1, 0.2] ---
python search_texture_uncorrelated.py --config original --mode erm --selection_method oracle
python search_texture_uncorrelated.py --config original --mode irm --selection_method oracle --n_trials 50
python search_texture_uncorrelated.py --config original --mode irm --selection_method train_domain_val --n_trials 50
python search_texture_uncorrelated.py --config original --mode irm --selection_method leave_one_domain_out --n_trials 50

echo --- Config: diverse e=[0.1, 0.5] ---
python search_texture_uncorrelated.py --config diverse --mode erm --selection_method oracle
python search_texture_uncorrelated.py --config diverse --mode irm --selection_method oracle --n_trials 50
python search_texture_uncorrelated.py --config diverse --mode irm --selection_method train_domain_val --n_trials 50
python search_texture_uncorrelated.py --config diverse --mode irm --selection_method leave_one_domain_out --n_trials 50

echo --- Config: proximate e=[0.7, 0.8] ---
python search_texture_uncorrelated.py --config proximate --mode erm --selection_method oracle
python search_texture_uncorrelated.py --config proximate --mode irm --selection_method oracle --n_trials 50
python search_texture_uncorrelated.py --config proximate --mode irm --selection_method train_domain_val --n_trials 50
python search_texture_uncorrelated.py --config proximate --mode irm --selection_method leave_one_domain_out --n_trials 50

echo All uncorrelated texture searches finished.