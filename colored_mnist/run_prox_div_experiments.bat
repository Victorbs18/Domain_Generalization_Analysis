@echo off

echo ============================================================
echo   Proximity vs Diversity 2x2 Final Experiments
echo ============================================================

echo --- Config A: Low diversity, Low proximity ---
python -u main_prox_div.py --config=A --selection_method=oracle --hidden_dim=102 --l2_regularizer_weight=0.0004610567249984422 --lr=0.00036628341624606644 --penalty_anneal_iters=221 --penalty_weight=33624.636376440816 --steps=501 --n_restarts=10
python -u main_prox_div.py --config=A --selection_method=train_domain_val --hidden_dim=237 --l2_regularizer_weight=0.00012174127103076253 --lr=0.000386945605970897 --penalty_anneal_iters=118 --penalty_weight=704029.1774325838 --steps=101 --n_restarts=10
python -u main_prox_div.py --config=A --selection_method=leave_one_domain_out --hidden_dim=95 --l2_regularizer_weight=0.00025128294378026306 --lr=0.0003497870814317026 --penalty_anneal_iters=159 --penalty_weight=53975.39297668061 --steps=101 --n_restarts=10

echo --- Config B: Low diversity, High proximity ---
python -u main_prox_div.py --config=B --selection_method=oracle --hidden_dim=437 --l2_regularizer_weight=1.1873492278636356e-05 --lr=0.0011355285091982394 --penalty_anneal_iters=79 --penalty_weight=265.2098286875673 --steps=401 --n_restarts=10
python -u main_prox_div.py --config=B --selection_method=train_domain_val --hidden_dim=253 --l2_regularizer_weight=0.00013212140597980882 --lr=0.0003930670759992006 --penalty_anneal_iters=197 --penalty_weight=1121.610493247227 --steps=401 --n_restarts=10
python -u main_prox_div.py --config=B --selection_method=leave_one_domain_out --hidden_dim=253 --l2_regularizer_weight=0.00013212140597980882 --lr=0.0003930670759992006 --penalty_anneal_iters=197 --penalty_weight=1121.610493247227 --steps=401 --n_restarts=10

echo --- Config C: High diversity, Low proximity ---
python -u main_prox_div.py --config=C --selection_method=oracle --hidden_dim=99 --l2_regularizer_weight=0.00016810705158715585 --lr=0.0022563323664207004 --penalty_anneal_iters=56 --penalty_weight=171.08461891527722 --steps=201 --n_restarts=10
python -u main_prox_div.py --config=C --selection_method=train_domain_val --hidden_dim=176 --l2_regularizer_weight=0.00029656574289180465 --lr=0.002740478851752145 --penalty_anneal_iters=82 --penalty_weight=7833.531681967299 --steps=301 --n_restarts=10
python -u main_prox_div.py --config=C --selection_method=leave_one_domain_out --hidden_dim=119 --l2_regularizer_weight=0.0002922483958866366 --lr=0.0014550040854539377 --penalty_anneal_iters=112 --penalty_weight=1398.894322376874 --steps=501 --n_restarts=10

echo --- Config D: High diversity, High proximity ---
python -u main_prox_div.py --config=D --selection_method=oracle --hidden_dim=215 --l2_regularizer_weight=0.0008903070341397012 --lr=0.000384100012359557 --penalty_anneal_iters=201 --penalty_weight=146.70897925451902 --steps=301 --n_restarts=10
python -u main_prox_div.py --config=D --selection_method=train_domain_val --hidden_dim=215 --l2_regularizer_weight=0.0008903070341397012 --lr=0.000384100012359557 --penalty_anneal_iters=201 --penalty_weight=146.70897925451902 --steps=301 --n_restarts=10
python -u main_prox_div.py --config=D --selection_method=leave_one_domain_out --hidden_dim=176 --l2_regularizer_weight=0.00029656574289180465 --lr=0.002740478851752145 --penalty_anneal_iters=82 --penalty_weight=7833.531681967299 --steps=301 --n_restarts=10

echo All proximity vs diversity experiments finished.