@echo off

echo ============================================================
echo   IRM - Train Domain Validation
echo ============================================================
python -u main.py --hidden_dim=486 --l2_regularizer_weight=0.0019205212018125557 --lr=0.0019322544016001175 --penalty_anneal_iters=152 --penalty_weight=6332.041485861786 --steps=101 --n_restarts=10

echo ============================================================
echo   IRM - Leave One Domain Out
echo ============================================================
python -u main.py --hidden_dim=80 --l2_regularizer_weight=0.005598034045897476 --lr=0.0005372964744550523 --penalty_anneal_iters=97 --penalty_weight=16573.33380333007 --steps=101 --n_restarts=10

echo ============================================================
echo   ERM - Train Domain Validation
echo ============================================================
python -u main.py --hidden_dim=256 --l2_regularizer_weight=0.001 --lr=0.001 --penalty_anneal_iters=0 --penalty_weight=0.0 --steps=501 --n_restarts=10

echo ============================================================
echo   ERM - Leave One Domain Out
echo ============================================================
python -u main.py --hidden_dim=256 --l2_regularizer_weight=0.001 --lr=0.001 --penalty_anneal_iters=0 --penalty_weight=0.0 --steps=501 --n_restarts=10

echo ============================================================
echo   Grayscale - Train Domain Validation
echo ============================================================
python -u main.py --grayscale_model --hidden_dim=88 --l2_regularizer_weight=0.0011198164932559394 --lr=0.0017569611396615358 --penalty_anneal_iters=0 --penalty_weight=0.0 --steps=201 --n_restarts=10

echo ============================================================
echo   Grayscale - Leave One Domain Out
echo ============================================================
python -u main.py --grayscale_model --hidden_dim=255 --l2_regularizer_weight=0.001304656360450818 --lr=0.0019086620171489411 --penalty_anneal_iters=0 --penalty_weight=0.0 --steps=401 --n_restarts=10

echo All realistic experiments finished.