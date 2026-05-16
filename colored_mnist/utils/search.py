# utils/search.py
# Hyperparameter sampling for Colored MNIST experiments.
#
# Search ranges replicate those described in the original IRM paper:
#   hidden_dim            = int(2 ** U(6, 9))      
#   l2_regularizer_weight = 10 ** U(-5, -2)         
#   lr                    = 10 ** U(-3.5, -2.5)     
#   penalty_anneal_iters  = randint(50, 250)        
#   penalty_weight        = 10 ** U(2, 6)         
#   steps                 in {101, 201, 301, 401, 501}

import numpy as np

STEPS_GRID = [101, 201, 301, 401, 501]


def sample_irm(rng: np.random.Generator) -> dict:
    """
    Sample a random IRM hyperparameter configuration.
    Uses log-uniform sampling for parameters that vary over
    orders of magnitude.
    """
    return {
        "hidden_dim":            int(2 ** rng.uniform(6, 9)),
        "l2_regularizer_weight": float(10 ** rng.uniform(-5, -2)),
        "lr":                    float(10 ** rng.uniform(-3.5, -2.5)),
        "penalty_anneal_iters":  int(rng.integers(50, 250)),
        "penalty_weight":        float(10 ** rng.uniform(2, 6)),
        "steps":                 int(rng.choice(STEPS_GRID)),
        "grayscale_model":       False,
    }


def sample_grayscale(rng: np.random.Generator) -> dict:
    """
    Sample a random grayscale ERM hyperparameter configuration.
    No IRM penalty terms.
    """
    return {
        "hidden_dim":            int(2 ** rng.uniform(6, 9)),
        "l2_regularizer_weight": float(10 ** rng.uniform(-5, -2)),
        "lr":                    float(10 ** rng.uniform(-3.5, -2.5)),
        "penalty_anneal_iters":  0,
        "penalty_weight":        0.0,
        "steps":                 int(rng.choice(STEPS_GRID)),
        "grayscale_model":       True,
    }


# Fixed ERM hyperparameters: no search needed.
# As noted in the original paper, searching for ERM hyperparameters is not meaningful because ERM performs worse than random guessing
# on this task and the search would find degenerate solutions.
ERM_FIXED = {
    "hidden_dim":            256,
    "l2_regularizer_weight": 0.001,
    "lr":                    0.001,
    "penalty_anneal_iters":  0,
    "penalty_weight":        0.0,
    "steps":                 501,
    "grayscale_model":       False,
}