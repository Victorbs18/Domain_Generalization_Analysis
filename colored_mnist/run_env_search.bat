@echo off

echo ============================================================
echo   Searching all n_envs x selection_method combinations
echo ============================================================

echo --- n_envs=1 ---
python search_env.py --n_envs 1 --selection_method oracle --n_trials 50
python search_env.py --n_envs 1 --selection_method train_domain_val --n_trials 50
python search_env.py --n_envs 1 --selection_method leave_one_domain_out --n_trials 50

echo --- n_envs=2 ---
python search_env.py --n_envs 2 --selection_method oracle --n_trials 50
python search_env.py --n_envs 2 --selection_method train_domain_val --n_trials 50
python search_env.py --n_envs 2 --selection_method leave_one_domain_out --n_trials 50

echo --- n_envs=3 ---
python search_env.py --n_envs 3 --selection_method oracle --n_trials 50
python search_env.py --n_envs 3 --selection_method train_domain_val --n_trials 50
python search_env.py --n_envs 3 --selection_method leave_one_domain_out --n_trials 50

echo --- n_envs=4 ---
python search_env.py --n_envs 4 --selection_method oracle --n_trials 50
python search_env.py --n_envs 4 --selection_method train_domain_val --n_trials 50
python search_env.py --n_envs 4 --selection_method leave_one_domain_out --n_trials 50

echo --- n_envs=5 ---
python search_env.py --n_envs 5 --selection_method oracle --n_trials 50
python search_env.py --n_envs 5 --selection_method train_domain_val --n_trials 50
python search_env.py --n_envs 5 --selection_method leave_one_domain_out --n_trials 50

echo All searches finished.