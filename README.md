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

## Experiment 5 — Proximity vs Diversity 2x2

Experiments 3 and 4 together showed that environment diversity is the key
factor for IRM to recover under realistic model selection — not quantity.
However, a confound remained: in Experiment 3, adding more diverse
environments also increased their proximity to the test distribution
(e=0.9). This experiment disentangles the two factors with a clean 2x2
design.

**Research question:** Is IRM's recovery under realistic model selection
driven by environment *diversity*, environment *proximity* to the test
distribution, or both?

### Design

Always n=2 training environments. Test environment always fixed at e=0.9.

| Config | e values | Diversity | Proximity to test (e=0.9) |
|--------|---------|-----------|--------------------------|
| A | {0.1, 0.2} | Low | Low |
| B | {0.7, 0.8} | Low | **High** |
| C | {0.1, 0.5} | **High** | Low |
| D | {0.1, 0.8} | **High** | **High** |

**Prediction logic:**
- If diversity is the key factor → C and D recover, A and B fail
- If proximity is the key factor → B and D recover, A and C fail
- If both matter equally → only D fully recovers

### How to run
```bash
# Step 1 — Find hyperparameters (12 combinations)
run_prox_div_search.bat > results_prox_div_search.txt 2>&1

# Step 2 — Run final experiments
run_prox_div_experiments.bat > results_prox_div_experiments.txt 2>&1
```

### Hyperparameters found

| Config | Selection | hidden_dim | lr | l2_reg | penalty_anneal_iters | penalty_weight | steps |
|--------|:---------:|:----------:|:--:|:------:|:--------------------:|:--------------:|:-----:|
| A | Oracle | 102 | 0.000366 | 0.000461 | 221 | 33624.6 | 501 |
| A | Train domain val | 237 | 0.000387 | 0.000122 | 118 | 704029.2 | 101 |
| A | Leave one domain out | 95 | 0.000350 | 0.000251 | 159 | 53975.4 | 101 |
| B | Oracle | 437 | 0.001136 | 1.19e-05 | 79 | 265.2 | 401 |
| B | Train domain val | 253 | 0.000393 | 0.000132 | 197 | 1121.6 | 401 |
| B | Leave one domain out | 253 | 0.000393 | 0.000132 | 197 | 1121.6 | 401 |
| C | Oracle | 99 | 0.002256 | 0.000168 | 56 | 171.1 | 201 |
| C | Train domain val | 176 | 0.002740 | 0.000297 | 82 | 7833.5 | 301 |
| C | Leave one domain out | 119 | 0.001455 | 0.000292 | 112 | 1398.9 | 501 |
| D | Oracle | 215 | 0.000384 | 0.000890 | 201 | 146.7 | 301 |
| D | Train domain val | 215 | 0.000384 | 0.000890 | 201 | 146.7 | 301 |
| D | Leave one domain out | 176 | 0.002740 | 0.000297 | 82 | 7833.5 | 301 |

### Results

| Config | e values | Diversity | Proximity | Oracle | Train Domain Val | Leave One Domain Out |
|--------|---------|:---------:|:---------:|:------:|:----------------:|:--------------------:|
| A | {0.1, 0.2} | Low | Low | 67.56% ± 0.68% | 10.59% ± 0.15% | 10.25% ± 0.08% |
| B | {0.7, 0.8} | Low | **High** | 81.69% ± 0.35% | **77.93% ± 0.13%** | **77.04% ± 0.27%** |
| C | {0.1, 0.5} | **High** | Low | 71.71% ± 0.19% | 69.44% ± 1.82% | 71.17% ± 0.23% |
| D | {0.1, 0.8} | **High** | **High** | 72.80% ± 0.06% | 72.58% ± 0.11% | 70.47% ± 5.68% |

---

## Experiment 6 — Two Correlated Spurious Features (Color + Texture)

Extends the Colored MNIST setup with a **second spurious feature**: a
stripe texture pattern applied to background pixels. Both spurious
features (color and texture) share the same correlation parameter e,
so they vary together across environments.

**Causal feature:** digit shape (invariant, always predicts the label)
**Spurious feature 1:** digit color (red/green, correlation = e)
**Spurious feature 2:** background stripe pattern (A/B, correlation = e)

This experiment directly tests whether IRM's ability to identify the
invariant feature degrades when two spurious correlations are present
simultaneously, and whether the diversity and proximity insights from
Experiments 3–5 still hold.

### Implementation

The stripe texture is applied exclusively to **background pixels**
(pixels with value 0 in the original MNIST image), preserving the
digit shape exactly. Two stripe patterns are used:

- **Pattern A:** even rows lit (correlated with label=0)
- **Pattern B:** odd rows lit (correlated with label=1)

The texture correlation strength equals the color correlation strength
`e` in all environments, so both spurious features are equally strong.

### Experimental design

Three environment configurations, mirroring Experiment 5:

| Config | e values | Diversity | Proximity to test (e=0.9) |
|--------|---------|-----------|--------------------------|
| original | {0.1, 0.2} | Low | Low |
| diverse | {0.1, 0.5} | High | Low |
| proximate | {0.7, 0.8} | Low | High |

### How to run
```bash
# Step 1 — Find hyperparameters
run_texture_search.bat > results_texture_search.txt 2>&1

# Step 2 — Run final experiments
run_texture_experiments.bat > results_texture_experiments.txt 2>&1
```

### Hyperparameters found

| Config | Mode | Selection | hidden_dim | lr | l2_reg | penalty_anneal_iters | penalty_weight | steps |
|--------|:----:|:---------:|:----------:|:--:|:------:|:--------------------:|:--------------:|:-----:|
| original | IRM | Oracle | 87 | 0.000883 | 0.001227 | 83 | 1607.1 | 201 |
| original | IRM | Train domain val | 215 | 0.000384 | 0.000890 | 201 | 146.7 | 301 |
| original | IRM | Leave one domain out | 254 | 0.002061 | 0.000166 | 114 | 123.3 | 101 |
| diverse | IRM | All methods | 437 | 0.001136 | 1.19e-05 | 79 | 265.2 | 401 |
| proximate | IRM | All methods | 340 | 0.001926 | 0.000252 | 219 | 273.3 | 101 |
| All | ERM | Oracle | 256 | 0.001 | 0.001 | 0 | 0.0 | 501 |

### Results

| Config | e values | Method | Oracle | Train Domain Val | Leave One Domain Out |
|--------|---------|:------:|:------:|:----------------:|:--------------------:|
| original | {0.1, 0.2} | ERM | 12.97% ± 0.15% | — | — |
| original | {0.1, 0.2} | IRM | 54.79% ± 5.71% | 34.13% ± 0.53% | 12.39% ± 0.88% |
| diverse | {0.1, 0.5} | ERM | 31.92% ± 0.66% | — | — |
| diverse | {0.1, 0.5} | IRM | 70.14% ± 0.33% | **70.54% ± 0.27%** | **70.14% ± 0.33%** |
| proximate | {0.7, 0.8} | ERM | **91.67% ± 0.55%** | — | — |
| proximate | {0.7, 0.8} | IRM | **90.60% ± 0.76%** | **90.70% ± 0.19%** | **90.60% ± 0.76%** |

### Comparison with single spurious feature (Experiment 5)

| Config | IRM oracle — 1 spurious | IRM oracle — 2 spurious | Δ |
|--------|:-----------------------:|:-----------------------:|:-:|
| original {0.1, 0.2} | 66.27% ± 1.44% | 54.79% ± 5.71% | **-11.5%** ↓ |
| diverse {0.1, 0.5} | 71.71% ± 0.19% | 70.14% ± 0.33% | -1.6% ≈ |
| proximate {0.7, 0.8} | 81.69% ± 0.35% | 90.60% ± 0.76% | **+8.9%** ↑ |

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
used. This isolates diversity as the active ingredient, not quantity.

**5. Proximity to the test distribution is the dominant factor** —
Experiment 5 directly disentangles diversity and proximity. Config B
({0.7, 0.8} — low diversity, high proximity) achieves 77-78% under
realistic selection, outperforming Config C ({0.1, 0.5} — high diversity,
low proximity) at ~69-71%. Proximity alone is sufficient and more powerful
than diversity alone.

**6. Diversity provides a secondary stabilizing effect** — Config C
recovers to ~71% with high diversity alone, confirming diversity
independently helps even without proximity. However the recovery is less
stable (std up to 1.82%) compared to proximity-based recovery (std
0.13-0.27% for Config B).

**7. The combined effect does not compound** — Config D ({0.1, 0.8} —
high diversity, high proximity) does not outperform Config B alone. Oracle
accuracy is actually lower for D (72.8%) than B (81.7%), suggesting the
two environments {0.1, 0.8} present a harder optimization problem despite
covering both factors.

**8. Hyperparameter sensitivity is the root cause of collapse** — The
`penalty_weight` hyperparameter varies by 4 orders of magnitude across
successful configs (146 to 704029). Realistic selection consistently finds
degenerate configs with very short training (steps=101) and extreme penalty
weights that collapse test accuracy, confirming that IRM's instability is
fundamentally a hyperparameter sensitivity problem. Proximity and diversity
help because they make the invariant solution easier to find and more
robustly identifiable by realistic selection criteria.

**9. Two spurious features degrade IRM under weak environments** — With
low diversity environments {0.1, 0.2}, adding a second spurious feature
(texture) causes a significant drop in IRM oracle accuracy from 66.3% to
54.8%, and increases variance substantially (std 1.4% → 5.7%). This
confirms the theoretical expectation that multiple spurious correlations
make the invariant feature harder to identify.

**10. Environment diversity makes IRM robust to multiple spurious
features** — With diverse environments {0.1, 0.5}, IRM oracle accuracy
drops only marginally (-1.6%) when a second spurious feature is added.
More strikingly, realistic selection fully recovers to ~70% across all
selection methods — matching oracle performance. Diversity not only helps
with model selection but also buffers against the added difficulty of
multiple spurious correlations.

**11. Proximity can make the problem trivially easy — for the wrong
reasons** — With proximate environments {0.7, 0.8}, ERM achieves 91.7%
test accuracy with two spurious features — higher than IRM oracle in the
single-spurious case. This is not because the model learned the invariant
feature, but because the test distribution (e=0.9) is so close to the
training environments that spurious features remain predictive at test
time. This is a cautionary result: high test accuracy does not imply
invariant learning, and proximity to the test distribution can mask
IRM's failure to generalize causally.

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
