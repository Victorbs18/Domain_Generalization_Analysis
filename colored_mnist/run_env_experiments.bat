@echo off

echo ============================================================
echo   n_envs=1
echo ============================================================
python -u main_env.py --n_envs=1 --selection_method=oracle --hidden_dim=258 --l2_regularizer_weight=6.977524201596243e-05 --lr=0.0014435194325372637 --penalty_anneal_iters=196 --penalty_weight=118738.06512263101 --steps=401 --n_restarts=10
python -u main_env.py --n_envs=1 --selection_method=train_domain_val --hidden_dim=70 --l2_regularizer_weight=0.00020197200649695645 --lr=0.0031072454736909326 --penalty_anneal_iters=80 --penalty_weight=98726.12193926235 --steps=501 --n_restarts=10
python -u main_env.py --n_envs=1 --selection_method=leave_one_domain_out --hidden_dim=486 --l2_regularizer_weight=0.0019205212018125557 --lr=0.0019322544016001175 --penalty_anneal_iters=152 --penalty_weight=6332.041485861786 --steps=101 --n_restarts=10

echo ============================================================
echo   n_envs=2
echo ============================================================
python -u main_env.py --n_envs=2 --selection_method=oracle --hidden_dim=85 --l2_regularizer_weight=0.00018086017967290882 --lr=0.0029257142847283393 --penalty_anneal_iters=83 --penalty_weight=539626.0247628883 --steps=301 --n_restarts=10
python -u main_env.py --n_envs=2 --selection_method=train_domain_val --hidden_dim=486 --l2_regularizer_weight=0.0019205212018125557 --lr=0.0019322544016001175 --penalty_anneal_iters=152 --penalty_weight=6332.041485861786 --steps=101 --n_restarts=10
python -u main_env.py --n_envs=2 --selection_method=leave_one_domain_out --hidden_dim=103 --l2_regularizer_weight=2.4858026883262884e-05 --lr=0.0015054234390854927 --penalty_anneal_iters=243 --penalty_weight=10600.33832171295 --steps=101 --n_restarts=10

echo ============================================================
echo   n_envs=3
echo ============================================================
python -u main_env.py --n_envs=3 --selection_method=oracle --hidden_dim=256 --l2_regularizer_weight=0.0002589933531580329 --lr=0.0011620802131636504 --penalty_anneal_iters=65 --penalty_weight=34583.84531216486 --steps=401 --n_restarts=10
python -u main_env.py --n_envs=3 --selection_method=train_domain_val --hidden_dim=117 --l2_regularizer_weight=0.00035089032014738784 --lr=0.0005701184432131363 --penalty_anneal_iters=248 --penalty_weight=455.44010437038105 --steps=501 --n_restarts=10
python -u main_env.py --n_envs=3 --selection_method=leave_one_domain_out --hidden_dim=319 --l2_regularizer_weight=0.00020731719263694523 --lr=0.0022834837212690044 --penalty_anneal_iters=67 --penalty_weight=238.0725871895354 --steps=401 --n_restarts=10

echo ============================================================
echo   n_envs=4
echo ============================================================
python -u main_env.py --n_envs=4 --selection_method=oracle --hidden_dim=102 --l2_regularizer_weight=0.0004610567249984422 --lr=0.00036628341624606644 --penalty_anneal_iters=221 --penalty_weight=33624.636376440816 --steps=501 --n_restarts=10
python -u main_env.py --n_envs=4 --selection_method=train_domain_val --hidden_dim=423 --l2_regularizer_weight=0.0012563811197821091 --lr=0.0005832704324891089 --penalty_anneal_iters=202 --penalty_weight=130317.76284459846 --steps=501 --n_restarts=10
python -u main_env.py --n_envs=4 --selection_method=leave_one_domain_out --hidden_dim=341 --l2_regularizer_weight=0.00014535672756798012 --lr=0.0006142258697148462 --penalty_anneal_iters=97 --penalty_weight=362.2512847791626 --steps=401 --n_restarts=10

echo ============================================================
echo   n_envs=5
echo ============================================================
python -u main_env.py --n_envs=5 --selection_method=oracle --hidden_dim=70 --l2_regularizer_weight=0.00020197200649695645 --lr=0.0031072454736909326 --penalty_anneal_iters=80 --penalty_weight=98726.12193926235 --steps=501 --n_restarts=10
python -u main_env.py --n_envs=5 --selection_method=train_domain_val --hidden_dim=117 --l2_regularizer_weight=0.00035089032014738784 --lr=0.0005701184432131363 --penalty_anneal_iters=248 --penalty_weight=455.44010437038105 --steps=501 --n_restarts=10
python -u main_env.py --n_envs=5 --selection_method=leave_one_domain_out --hidden_dim=253 --l2_regularizer_weight=0.00013212140597980882 --lr=0.0003930670759992006 --penalty_anneal_iters=197 --penalty_weight=1121.610493247227 --steps=401 --n_restarts=10

echo All environment experiments finished.