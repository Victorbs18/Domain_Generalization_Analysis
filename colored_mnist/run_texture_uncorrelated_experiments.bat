@echo off

echo ============================================================
echo   Two INDEPENDENT Spurious Features — Final Experiments
echo ============================================================

echo --- Config: original e=[0.1, 0.2] ---
python -u main_texture_uncorrelated.py --config=original --mode=erm --selection_method=oracle --hidden_dim=256 --l2_regularizer_weight=0.001 --lr=0.001 --penalty_anneal_iters=0 --penalty_weight=0.0 --steps=501 --n_restarts=10
python -u main_texture_uncorrelated.py --config=original --mode=irm --selection_method=oracle --hidden_dim=423 --l2_regularizer_weight=0.0012563811197821091 --lr=0.0005832704324891089 --penalty_anneal_iters=202 --penalty_weight=130317.76284459846 --steps=501 --n_restarts=10
python -u main_texture_uncorrelated.py --config=original --mode=irm --selection_method=train_domain_val --hidden_dim=254 --l2_regularizer_weight=0.0001656382312621769 --lr=0.0020607266362084172 --penalty_anneal_iters=114 --penalty_weight=123.26741610799746 --steps=101 --n_restarts=10
python -u main_texture_uncorrelated.py --config=original --mode=irm --selection_method=leave_one_domain_out --hidden_dim=254 --l2_regularizer_weight=0.0001656382312621769 --lr=0.0020607266362084172 --penalty_anneal_iters=114 --penalty_weight=123.26741610799746 --steps=101 --n_restarts=10

echo --- Config: diverse e=[0.1, 0.5] ---
python -u main_texture_uncorrelated.py --config=diverse --mode=erm --selection_method=oracle --hidden_dim=256 --l2_regularizer_weight=0.001 --lr=0.001 --penalty_anneal_iters=0 --penalty_weight=0.0 --steps=501 --n_restarts=10
python -u main_texture_uncorrelated.py --config=diverse --mode=irm --selection_method=oracle --hidden_dim=437 --l2_regularizer_weight=1.1873492278636356e-05 --lr=0.0011355285091982394 --penalty_anneal_iters=79 --penalty_weight=265.2098286875673 --steps=401 --n_restarts=10
python -u main_texture_uncorrelated.py --config=diverse --mode=irm --selection_method=train_domain_val --hidden_dim=437 --l2_regularizer_weight=1.1873492278636356e-05 --lr=0.0011355285091982394 --penalty_anneal_iters=79 --penalty_weight=265.2098286875673 --steps=401 --n_restarts=10
python -u main_texture_uncorrelated.py --config=diverse --mode=irm --selection_method=leave_one_domain_out --hidden_dim=111 --l2_regularizer_weight=9.878863667554657e-05 --lr=0.0010487510350832606 --penalty_anneal_iters=219 --penalty_weight=122.02486041068684 --steps=301 --n_restarts=10

echo --- Config: proximate e=[0.7, 0.8] ---
python -u main_texture_uncorrelated.py --config=proximate --mode=erm --selection_method=oracle --hidden_dim=256 --l2_regularizer_weight=0.001 --lr=0.001 --penalty_anneal_iters=0 --penalty_weight=0.0 --steps=501 --n_restarts=10
python -u main_texture_uncorrelated.py --config=proximate --mode=irm --selection_method=oracle --hidden_dim=340 --l2_regularizer_weight=0.0002524322304507403 --lr=0.001926475318213037 --penalty_anneal_iters=219 --penalty_weight=273.2599515914586 --steps=101 --n_restarts=10
python -u main_texture_uncorrelated.py --config=proximate --mode=irm --selection_method=train_domain_val --hidden_dim=103 --l2_regularizer_weight=2.4858026883262884e-05 --lr=0.0015054234390854927 --penalty_anneal_iters=243 --penalty_weight=10600.33832171295 --steps=101 --n_restarts=10
python -u main_texture_uncorrelated.py --config=proximate --mode=irm --selection_method=leave_one_domain_out --hidden_dim=340 --l2_regularizer_weight=0.0002524322304507403 --lr=0.001926475318213037 --penalty_anneal_iters=219 --penalty_weight=273.2599515914586 --steps=101 --n_restarts=10

echo All uncorrelated texture experiments finished.