# Domain Generalization Analysis
# Invariant Risk Minimization — Reproduction & Analysis

Reproduction and extension of the **Colored MNIST** experiment from:

> Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019).
> *Invariant Risk Minimization.*
> https://arxiv.org/abs/1907.02893

With additional experiments inspired by:

> Gulrajani, I., & Lopez-Paz, D. (2021).
> *In Search of Lost Domain Generalization.*
> https://arxiv.org/abs/2007.01434

---

## Overview

This repository reproduces the Colored MNIST IRM experiments and extends
them along two original research directions:

1. **Model selection methods** — does IRM's advantage hold under realistic
   (non-oracle) hyperparameter selection?
2. **Number of environments** — how many training environments does IRM need,
   and does diversity matter more than quantity?

---

## Repository Structure
```
Domain_Generalization_Analysis/
├── environment.yaml
├── README.md
└── colored_mnist/
    │
    ├── main.py                      # Original IRM reproduction
    ├── search.py                    # Oracle hyperparameter search
    ├── run_experiments.bat          # Run original 3 conditions
    │
    ├── main_realistic.py            # Realistic model selection data splits
    ├── search_realistic.py          # Search with selection_method flag
    ├── run_realistic.bat            # Run realistic selection experiments
    │
    ├── main_env.py                  # Variable environments (increasing diversity)
    ├── search_env.py                # Search for variable environments
    ├── run_env_search.bat           # Run all environment searches
    ├── run_env_experiments.bat      # Run all environment experiments
    │
    ├── main_similar_env.py          # Variable environments (fixed range)
    ├── search_similar_env.py        # Search for fixed range environments
    ├── run_similar_env_search.bat   # Run fixed range searches
    └── run_similar_env_experiments.bat  # Run fixed range experiments
```

---

## Setup
```bash
conda env create -f environment.yaml
conda activate irm_reproduction
```

---

## Experiment 1 — IRM Reproduction

Reproduces the three conditions from the original paper under **oracle
model selection** (test environment used for hyperparameter selection).

### How to run
```bash
# Step 1 — Find hyperparameters
python search.py --mode irm --n_trials 50
python search.py --mode grayscale --n_trials 50
python search.py --mode erm

# Step 2 — Run final experiments
run_experiments.bat > results.txt 2>&1
```

### Hyperparameters found

| Method | hidden_dim | lr | l2_reg | penalty_anneal_iters | penalty_weight | steps |
|--------|:----------:|:--:|:------:|:--------------------:|:--------------:|:-----:|
| IRM | 256 | 0.00116 | 0.000259 | 65 | 34583.8 | 401 |
| ERM | 256 | 0.001 | 0.001 | 0 | 0.0 | 501 |
| Grayscale | 255 | 0.00191 | 0.001305 | 0 | 0.0 | 401 |

### Results

| Method | Test Accuracy | Paper (reported) |
|--------|:-------------:|:----------------:|
| ERM | 17.50% ± 0.51% | ~17% |
| IRM | 66.27% ± 1.44% | ~70% |
| Grayscale ERM (oracle) | 73.21% ± 0.19% | ~73% |

---

## Experiment 2 — Model Selection Methods

Tests whether IRM's advantage holds under **realistic model selection**,
where the test environment is never used for hyperparameter tuning.

Three selection strategies:

| Strategy | Validation set used |
|----------|-------------------|
| **Oracle** | Test env (e=0.9) — unrealistic upper bound |
| **Train domain val** | Held-out 20% of each train env |
| **Leave one domain out** | Each train env as val for the other |

### How to run
```bash
# Step 1 — Find hyperparameters
python search_realistic.py --mode irm --selection_method train_domain_val --n_trials 50
python search_realistic.py --mode irm --selection_method leave_one_domain_out --n_trials 50
python search_realistic.py --mode erm --selection_method train_domain_val
python search_realistic.py --mode erm --selection_method leave_one_domain_out
python search_realistic.py --mode grayscale --selection_method train_domain_val --n_trials 50
python search_realistic.py --mode grayscale --selection_method leave_one_domain_out --n_trials 50

# Step 2 — Run final experiments
run_realistic.bat > results_realistic.txt 2>&1
```

### Hyperparameters found

| Method | Selection | hidden_dim | lr | l2_reg | penalty_anneal_iters | penalty_weight | steps |
|--------|:---------:|:----------:|:--:|:------:|:--------------------:|:--------------:|:-----:|
| IRM | Oracle | 256 | 0.00116 | 0.000259 | 65 | 34583.8 | 401 |
| IRM | Train domain val | 486 | 0.00193 | 0.001921 | 152 | 6332.0 | 101 |
| IRM | Leave one domain out | 80 | 0.000537 | 0.005598 | 97 | 16573.3 | 101 |
| ERM | All methods | 256 | 0.001 | 0.001 | 0 | 0.0 | 501 |
| Grayscale | Train domain val | 88 | 0.00176 | 0.001120 | 0 | 0.0 | 201 |
| Grayscale | Leave one domain out | 255 | 0.00191 | 0.001305 | 0 | 0.0 | 401 |

### Results

| Method | Oracle | Train Domain Val | Leave One Domain Out |
|--------|:------:|:----------------:|:--------------------:|
| ERM | 17.50% ± 0.51% | 17.50% ± 0.51% | 17.50% ± 0.51% |
| IRM | 66.27% ± 1.44% | 11.26% ± 0.47% | 9.87% ± 0.00% |
| Grayscale ERM | 73.21% ± 0.19% | 73.33% ± 0.07% | 73.21% ± 0.19% |

---

## Experiment 3 — Number of Environments (Increasing Diversity)

Tests whether more training environments help IRM recover under realistic
model selection. Environments use **increasing e values** so both quantity
and diversity increase together.

| n_envs | e values |
|--------|---------|
| 1 | {0.1} |
| 2 | {0.1, 0.2} |
| 3 | {0.1, 0.2, 0.3} |
| 4 | {0.1, 0.2, 0.3, 0.4} |
| 5 | {0.1, 0.2, 0.3, 0.4, 0.45} |

Test environment always fixed at e=0.9.

### How to run
```bash
# Step 1 — Find hyperparameters (all 15 combinations)
run_env_search.bat > results_env_search.txt 2>&1

# Step 2 — Run final experiments
run_env_experiments.bat > results_env_experiments.txt 2>&1
```

### Hyperparameters found

| n_envs | Selection | hidden_dim | lr | l2_reg | penalty_anneal_iters | penalty_weight | steps |
|--------|:---------:|:----------:|:--:|:------:|:--------------------:|:--------------:|:-----:|
| 1 | Oracle | 258 | 0.001444 | 6.98e-05 | 196 | 118738.1 | 401 |
| 1 | Train domain val | 70 | 0.003107 | 0.000202 | 80 | 98726.1 | 501 |
| 1 | Leave one domain out | 486 | 0.001932 | 0.001921 | 152 | 6332.0 | 101 |
| 2 | Oracle | 85 | 0.002926 | 0.000181 | 83 | 539626.0 | 301 |
| 2 | Train domain val | 486 | 0.001932 | 0.001921 | 152 | 6332.0 | 101 |
| 2 | Leave one domain out | 103 | 0.001505 | 2.49e-05 | 243 | 10600.3 | 101 |
| 3 | Oracle | 256 | 0.001162 | 0.000259 | 65 | 34583.8 | 401 |
| 3 | Train domain val | 117 | 0.000570 | 0.000351 | 248 | 455.4 | 501 |
| 3 | Leave one domain out | 319 | 0.002283 | 0.000207 | 67 | 238.1 | 401 |
| 4 | Oracle | 102 | 0.000366 | 0.000461 | 221 | 33624.6 | 501 |
| 4 | Train domain val | 423 | 0.000583 | 0.001256 | 202 | 130317.8 | 501 |
| 4 | Leave one domain out | 341 | 0.000614 | 0.000145 | 97 | 362.3 | 401 |
| 5 | Oracle | 70 | 0.003107 | 0.000202 | 80 | 98726.1 | 501 |
| 5 | Train domain val | 117 | 0.000570 | 0.000351 | 248 | 455.4 | 501 |
| 5 | Leave one domain out | 253 | 0.000393 | 0.000132 | 197 | 1121.6 | 401 |

### Results

| n_envs | e values | Oracle | Train Domain Val | Leave One Domain Out |
|--------|---------|:------:|:----------------:|:--------------------:|
| 1 | {0.1} | 27.61% ± 10.81% | 9.90% ± 0.05% | 10.19% ± 0.00% |
| 2 | {0.1, 0.2} | 67.50% ± 1.78% | 10.70% ± 0.62% | 11.20% ± 0.31% |
| 3 | {0.1, 0.2, 0.3} | 68.33% ± 0.94% | 62.44% ± 0.76% | 54.50% ± 3.91% |
| 4 | {0.1, 0.2, 0.3, 0.4} | 70.47% ± 0.20% | 68.02% ± 1.57% | 67.14% ± 0.34% |
| 5 | {0.1, 0.2, 0.3, 0.4, 0.45} | 69.71% ± 2.75% | 66.66% ± 0.36% | 68.44% ± 0.42% |

---

## Experiment 4 — Number of Environments (Fixed Range {0.1, 0.2})

Isolates the **pure quantity effect** by keeping all e values within the
same fixed range {0.1, 0.2}. This controls for the confound in Experiment
3 where adding environments also increased proximity to the test
distribution (e=0.9).

| n_envs | e values |
|--------|---------|
| 1 | {0.15} |
| 2 | {0.1, 0.2} |
| 3 | {0.1, 0.15, 0.2} |
| 4 | {0.1, 0.133, 0.167, 0.2} |
| 5 | {0.1, 0.125, 0.15, 0.175, 0.2} |

Test environment always fixed at e=0.9.

### How to run
```bash
# Step 1 — Find hyperparameters (all 15 combinations)
run_similar_env_search.bat > results_similar_env_search.txt 2>&1

# Step 2 — Run final experiments
run_similar_env_experiments.bat > results_similar_env_experiments.txt 2>&1
```

### Hyperparameters found

| n_envs | Selection | hidden_dim | lr | l2_reg | penalty_anneal_iters | penalty_weight | steps |
|--------|:---------:|:----------:|:--:|:------:|:--------------------:|:--------------:|:-----:|
| 1 | Oracle | 309 | 0.002956 | 0.000116 | 139 | 129877.5 | 501 |
| 1 | Train domain val | 87 | 0.000883 | 0.001227 | 83 | 1607.1 | 201 |
| 1 | Leave one domain out | 92 | 0.002369 | 0.000628 | 58 | 1743.0 | 101 |
| 2 | Oracle | 114 | 0.001452 | 7.60e-05 | 75 | 136644.7 | 301 |
| 2 | Train domain val | 178 | 0.000441 | 9.76e-05 | 185 | 22417.0 | 101 |
| 2 | Leave one domain out | 77 | 0.000916 | 0.001469 | 193 | 10096.7 | 101 |
| 3 | Oracle | 114 | 0.001452 | 7.60e-05 | 75 | 136644.7 | 301 |
| 3 | Train domain val | 138 | 0.002034 | 0.003086 | 242 | 648033.7 | 201 |
| 3 | Leave one domain out | 95 | 0.000350 | 0.000251 | 159 | 53975.4 | 101 |
| 4 | Oracle | 85 | 0.002926 | 0.000181 | 83 | 539626.0 | 301 |
| 4 | Train domain val | 237 | 0.000387 | 0.000122 | 118 | 704029.2 | 101 |
| 4 | Leave one domain out | 486 | 0.001932 | 0.001921 | 152 | 6332.0 | 101 |
| 5 | Oracle | 114 | 0.001452 | 7.60e-05 | 75 | 136644.7 | 301 |
| 5 | Train domain val | 340 | 0.001926 | 0.000252 | 219 | 273.3 | 101 |
| 5 | Leave one domain out | 301 | 0.000670 | 0.007990 | 231 | 7554.8 | 201 |

### Results

| n_envs | e values | Oracle | Train Domain Val | Leave One Domain Out |
|--------|---------|:------:|:----------------:|:--------------------:|
| 1 | {0.15} | 43.08% ± 9.67% | 9.91% ± 0.03% | 10.55% ± 0.20% |
| 2 | {0.1, 0.2} | 68.41% ± 1.41% | 10.63% ± 0.13% | 10.23% ± 0.03% |
| 3 | {0.1, 0.15, 0.2} | 67.88% ± 0.72% | 10.04% ± 0.04% | 10.27% ± 0.11% |
| 4 | {0.1, 0.133, 0.167, 0.2} | 66.31% ± 1.53% | 10.24% ± 0.08% | 10.56% ± 0.24% |
| 5 | {0.1, 0.125, 0.15, 0.175, 0.2} | 65.43% ± 0.67% | 15.06% ± 1.50% | 10.23% ± 0.01% |

---

## Key Findings

**1. IRM reproduces the paper closely under oracle selection** — achieving
66.3% test accuracy vs the reported ~70%, with ERM at 17.5% and the
grayscale oracle at 73.2%.

**2. IRM collapses under realistic model selection** — dropping from 66%
to ~10% (worse than random chance) when the test environment is not used
for hyperparameter selection. ERM and the grayscale oracle are unaffected.
This directly reproduces the finding of *In Search of Lost Domain
Generalization*.

**3. Environment diversity recovers IRM under realistic selection** —
Experiment 3 shows that with 3+ environments spanning a wider e range,
realistic selection recovers to ~62-68%. At n=4,5 all three selection
methods converge to ~67-70%, matching oracle performance.

**4. Environment quantity alone does not help** — Experiment 4 shows that
adding more environments within the fixed range {0.1, 0.2} provides no
benefit under realistic selection, regardless of how many environments are
used. Test accuracy stays at ~10% for all n_envs under train domain val
and leave one domain out.

**5. The key factor is diversity, not quantity** — Comparing Experiments 3
and 4 directly isolates the confound: it is the spread of e values across
environments, not their number, that determines whether IRM can be reliably
selected without oracle access to the test distribution.

**6. Hyperparameter sensitivity is the root cause** — The `penalty_weight`
hyperparameter varies by 4 orders of magnitude across successful configs
(238 to 704029). Realistic selection consistently finds degenerate configs
with very short training (steps=101) and high penalty weights that collapse
test accuracy, confirming that IRM's instability is fundamentally a
hyperparameter sensitivity problem.

---

## Notes

- All experiments use 10 restarts, results reported as mean ± std
- Hyperparameters selected via 50-trial random search using the same
  ranges as the original paper
- Selection criterion: `max min(val_acc_env0, ..., val_acc_envN)`
- Fixed architecture: 3-layer MLP with ReLU activations
- Fixed optimizer: Adam
- Fixed loss: Binary Cross-Entropy
- Test environment always fixed at e=0.9

---

## References

1. Arjovsky et al. (2019). *Invariant Risk Minimization.*
   https://arxiv.org/abs/1907.02893

2. Gulrajani & Lopez-Paz (2021). *In Search of Lost Domain Generalization.*
   https://arxiv.org/abs/2007.01434
