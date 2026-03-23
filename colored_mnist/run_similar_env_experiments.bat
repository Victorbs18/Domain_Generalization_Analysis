@echo off

echo ============================================================
echo   Similar Env Range Experiments — fixed e in {0.1, 0.2}
echo ============================================================

echo --- n_envs=1 ---
python -u main_similar_env.py --n_envs=1 --selection_method=oracle --hidden_dim=309 --l2_regularizer_weight=0.00011576516747459546 --lr=0.0029559564028317454 --penalty_anneal_iters=139 --penalty_weight=129877.52032221333 --steps=501 --n_restarts=10
python -u main_similar_env.py --n_envs=1 --selection_method=train_domain_val --hidden_dim=87 --l2_regularizer_weight=0.0012273293649724259 --lr=0.0008833977231323628 --penalty_anneal_iters=83 --penalty_weight=1607.120188116246 --steps=201 --n_restarts=10
python -u main_similar_env.py --n_envs=1 --selection_method=leave_one_domain_out --hidden_dim=92 --l2_regularizer_weight=0.0006282729685044822 --lr=0.0023689835271142007 --penalty_anneal_iters=58 --penalty_weight=1742.989181859607 --steps=101 --n_restarts=10

echo --- n_envs=2 ---
python -u main_similar_env.py --n_envs=2 --selection_method=oracle --hidden_dim=114 --l2_regularizer_weight=7.599434450298693e-05 --lr=0.0014518325021749573 --penalty_anneal_iters=75 --penalty_weight=136644.71413649357 --steps=301 --n_restarts=10
python -u main_similar_env.py --n_envs=2 --selection_method=train_domain_val --hidden_dim=178 --l2_regularizer_weight=9.763007816062847e-05 --lr=0.00044108692962177153 --penalty_anneal_iters=185 --penalty_weight=22417.04113333284 --steps=101 --n_restarts=10
python -u main_similar_env.py --n_envs=2 --selection_method=leave_one_domain_out --hidden_dim=77 --l2_regularizer_weight=0.001469190298101179 --lr=0.0009159615223789559 --penalty_anneal_iters=193 --penalty_weight=10096.691816860124 --steps=101 --n_restarts=10

echo --- n_envs=3 ---
python -u main_similar_env.py --n_envs=3 --selection_method=oracle --hidden_dim=114 --l2_regularizer_weight=7.599434450298693e-05 --lr=0.0014518325021749573 --penalty_anneal_iters=75 --penalty_weight=136644.71413649357 --steps=301 --n_restarts=10
python -u main_similar_env.py --n_envs=3 --selection_method=train_domain_val --hidden_dim=138 --l2_regularizer_weight=0.0030858103360688173 --lr=0.0020335341592305533 --penalty_anneal_iters=242 --penalty_weight=648033.6835889794 --steps=201 --n_restarts=10
python -u main_similar_env.py --n_envs=3 --selection_method=leave_one_domain_out --hidden_dim=95 --l2_regularizer_weight=0.00025128294378026306 --lr=0.0003497870814317026 --penalty_anneal_iters=159 --penalty_weight=53975.39297668061 --steps=101 --n_restarts=10

echo --- n_envs=4 ---
python -u main_similar_env.py --n_envs=4 --selection_method=oracle --hidden_dim=85 --l2_regularizer_weight=0.00018086017967290882 --lr=0.0029257142847283393 --penalty_anneal_iters=83 --penalty_weight=539626.0247628883 --steps=301 --n_restarts=10
python -u main_similar_env.py --n_envs=4 --selection_method=train_domain_val --hidden_dim=237 --l2_regularizer_weight=0.00012174127103076253 --lr=0.000386945605970897 --penalty_anneal_iters=118 --penalty_weight=704029.1774325838 --steps=101 --n_restarts=10
python -u main_similar_env.py --n_envs=4 --selection_method=leave_one_domain_out --hidden_dim=486 --l2_regularizer_weight=0.0019205212018125557 --lr=0.0019322544016001175 --penalty_anneal_iters=152 --penalty_weight=6332.041485861786 --steps=101 --n_restarts=10

echo --- n_envs=5 ---
python -u main_similar_env.py --n_envs=5 --selection_method=oracle --hidden_dim=114 --l2_regularizer_weight=7.599434450298693e-05 --lr=0.0014518325021749573 --penalty_anneal_iters=75 --penalty_weight=136644.71413649357 --steps=301 --n_restarts=10
python -u main_similar_env.py --n_envs=5 --selection_method=train_domain_val --hidden_dim=340 --l2_regularizer_weight=0.0002524322304507403 --lr=0.001926475318213037 --penalty_anneal_iters=219 --penalty_weight=273.2599515914586 --steps=101 --n_restarts=10
python -u main_similar_env.py --n_envs=5 --selection_method=leave_one_domain_out --hidden_dim=301 --l2_regularizer_weight=0.00798967653877021 --lr=0.0006696152843824843 --penalty_anneal_iters=231 --penalty_weight=7554.804756926672 --steps=201 --n_restarts=10

echo All similar env experiments finished.