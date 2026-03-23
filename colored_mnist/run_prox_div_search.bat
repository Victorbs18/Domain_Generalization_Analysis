@echo off

echo ============================================================
echo   Proximity vs Diversity 2x2 Search
echo ============================================================
echo   A: {0.1, 0.2} Low diversity,  Low proximity
echo   B: {0.7, 0.8} Low diversity,  High proximity
echo   C: {0.1, 0.5} High diversity, Low proximity
echo   D: {0.1, 0.8} High diversity, High proximity
echo ============================================================

echo --- Config A: Low diversity, Low proximity ---
python search_prox_div.py --config A --selection_method oracle --n_trials 50
python search_prox_div.py --config A --selection_method train_domain_val --n_trials 50
python search_prox_div.py --config A --selection_method leave_one_domain_out --n_trials 50

echo --- Config B: Low diversity, High proximity ---
python search_prox_div.py --config B --selection_method oracle --n_trials 50
python search_prox_div.py --config B --selection_method train_domain_val --n_trials 50
python search_prox_div.py --config B --selection_method leave_one_domain_out --n_trials 50

echo --- Config C: High diversity, Low proximity ---
python search_prox_div.py --config C --selection_method oracle --n_trials 50
python search_prox_div.py --config C --selection_method train_domain_val --n_trials 50
python search_prox_div.py --config C --selection_method leave_one_domain_out --n_trials 50

echo --- Config D: High diversity, High proximity ---
python search_prox_div.py --config D --selection_method oracle --n_trials 50
python search_prox_div.py --config D --selection_method train_domain_val --n_trials 50
python search_prox_div.py --config D --selection_method leave_one_domain_out --n_trials 50

echo All searches finished.