# Missing artifacts for the core-set metric

Every cell listed below could not be filled because the matching transport plan or heldout-subset file is absent from the local clone. To fill these cells, run the corresponding experiment with the patched `code/run_experiments.py` on a machine that has the av-bridge image / audio / caption embeddings.

The patch additions (already applied in this branch):

- `exp_c_transitive` now saves every (K, alpha) plan to `results/exp_c/plans/T__K{K}__a{alpha:.2f}.npy`
- `exp_d_caption` now saves every alpha plan to `results/exp_d/plans/T__a{alpha:.2f}.npy`
- all five image-audio experiments now save the heldout row-index subset to `results/exp_<x>/heldout_compare_idx.npy`

After re-running, re-run this script to regenerate the table:

```bash
python code/run_experiments.py --experiment c
python code/run_experiments.py --experiment d
python code/run_experiments.py --experiment unsup
python code/run_experiments.py --experiment text
python code/run_experiments.py --experiment random
python code/extended_metrics_table.py
```

## Missing cells, by method

### GW (unsup, alpha=1.0)

| K | alpha | scope | reason |
|---|---:|---|---|
| 0 | 1.0 | heldout_like_c | heldout subset index not found: results/exp_unsup/heldout_compare_idx.npy |

### FGW direct (M = caption cos)

| K | alpha | scope | reason |
|---|---:|---|---|
| 0 | 0.0 | aggregate | plan file not found: results/exp_d/plans/T__a0.00.npy |
| 0 | 0.3 | aggregate | plan file not found: results/exp_d/plans/T__a0.30.npy |
| 0 | 0.5 | aggregate | plan file not found: results/exp_d/plans/T__a0.50.npy |
| 0 | 0.7 | heldout_like_c | heldout subset index not found: results/exp_d/heldout_compare_idx.npy |
| 0 | 0.9 | aggregate | plan file not found: results/exp_d/plans/T__a0.90.npy |

### FGW transitive (identity bridge)

| K | alpha | scope | reason |
|---|---:|---|---|
| 10 | 0.0 | aggregate | plan file not found: results/exp_c/plans/T__K10__a0.00.npy |
| 10 | 0.0 | heldout_like_c | plan file not found: results/exp_c/plans/T__K10__a0.00.npy |
| 10 | 0.3 | aggregate | plan file not found: results/exp_c/plans/T__K10__a0.30.npy |
| 10 | 0.3 | heldout_like_c | plan file not found: results/exp_c/plans/T__K10__a0.30.npy |
| 10 | 0.5 | aggregate | plan file not found: results/exp_c/plans/T__K10__a0.50.npy |
| 10 | 0.5 | heldout_like_c | plan file not found: results/exp_c/plans/T__K10__a0.50.npy |
| 10 | 0.7 | aggregate | plan file not found: results/exp_c/plans/T__K10__a0.70.npy |
| 10 | 0.7 | heldout_like_c | plan file not found: results/exp_c/plans/T__K10__a0.70.npy |
| 10 | 0.9 | aggregate | plan file not found: results/exp_c/plans/T__K10__a0.90.npy |
| 10 | 0.9 | heldout_like_c | plan file not found: results/exp_c/plans/T__K10__a0.90.npy |
| 20 | 0.0 | aggregate | plan file not found: results/exp_c/plans/T__K20__a0.00.npy |
| 20 | 0.0 | heldout_like_c | plan file not found: results/exp_c/plans/T__K20__a0.00.npy |
| 20 | 0.3 | aggregate | plan file not found: results/exp_c/plans/T__K20__a0.30.npy |
| 20 | 0.3 | heldout_like_c | plan file not found: results/exp_c/plans/T__K20__a0.30.npy |
| 20 | 0.5 | aggregate | plan file not found: results/exp_c/plans/T__K20__a0.50.npy |
| 20 | 0.5 | heldout_like_c | plan file not found: results/exp_c/plans/T__K20__a0.50.npy |
| 20 | 0.7 | aggregate | plan file not found: results/exp_c/plans/T__K20__a0.70.npy |
| 20 | 0.7 | heldout_like_c | plan file not found: results/exp_c/plans/T__K20__a0.70.npy |
| 20 | 0.9 | aggregate | plan file not found: results/exp_c/plans/T__K20__a0.90.npy |
| 20 | 0.9 | heldout_like_c | plan file not found: results/exp_c/plans/T__K20__a0.90.npy |
| 50 | 0.0 | aggregate | plan file not found: results/exp_c/plans/T__K50__a0.00.npy |
| 50 | 0.0 | heldout_like_c | plan file not found: results/exp_c/plans/T__K50__a0.00.npy |
| 50 | 0.3 | aggregate | plan file not found: results/exp_c/plans/T__K50__a0.30.npy |
| 50 | 0.3 | heldout_like_c | plan file not found: results/exp_c/plans/T__K50__a0.30.npy |
| 50 | 0.5 | aggregate | plan file not found: results/exp_c/plans/T__K50__a0.50.npy |
| 50 | 0.5 | heldout_like_c | plan file not found: results/exp_c/plans/T__K50__a0.50.npy |
| 50 | 0.7 | aggregate | plan file not found: results/exp_c/plans/T__K50__a0.70.npy |
| 50 | 0.7 | heldout_like_c | plan file not found: results/exp_c/plans/T__K50__a0.70.npy |
| 50 | 0.9 | aggregate | plan file not found: results/exp_c/plans/T__K50__a0.90.npy |
| 50 | 0.9 | heldout_like_c | plan file not found: results/exp_c/plans/T__K50__a0.90.npy |
| 100 | 0.0 | aggregate | plan file not found: results/exp_c/plans/T__K100__a0.00.npy |
| 100 | 0.0 | heldout_like_c | plan file not found: results/exp_c/plans/T__K100__a0.00.npy |
| 100 | 0.3 | aggregate | plan file not found: results/exp_c/plans/T__K100__a0.30.npy |
| 100 | 0.3 | heldout_like_c | plan file not found: results/exp_c/plans/T__K100__a0.30.npy |
| 100 | 0.5 | aggregate | plan file not found: results/exp_c/plans/T__K100__a0.50.npy |
| 100 | 0.5 | heldout_like_c | plan file not found: results/exp_c/plans/T__K100__a0.50.npy |
| 100 | 0.7 | aggregate | plan file not found: results/exp_c/plans/T__K100__a0.70.npy |
| 100 | 0.7 | heldout_like_c | plan file not found: results/exp_c/plans/T__K100__a0.70.npy |
| 100 | 0.9 | aggregate | plan file not found: results/exp_c/plans/T__K100__a0.90.npy |
| 100 | 0.9 | heldout_like_c | plan file not found: results/exp_c/plans/T__K100__a0.90.npy |
| 160 | 0.0 | aggregate | plan file not found: results/exp_c/plans/T__K160__a0.00.npy |
| 160 | 0.0 | heldout_like_c | plan file not found: results/exp_c/plans/T__K160__a0.00.npy |
| 160 | 0.3 | aggregate | plan file not found: results/exp_c/plans/T__K160__a0.30.npy |
| 160 | 0.3 | heldout_like_c | plan file not found: results/exp_c/plans/T__K160__a0.30.npy |
| 160 | 0.5 | aggregate | plan file not found: results/exp_c/plans/T__K160__a0.50.npy |
| 160 | 0.5 | heldout_like_c | plan file not found: results/exp_c/plans/T__K160__a0.50.npy |
| 160 | 0.7 | aggregate | plan file not found: results/exp_c/plans/T__K160__a0.70.npy |
| 160 | 0.7 | heldout_like_c | plan file not found: results/exp_c/plans/T__K160__a0.70.npy |
| 160 | 0.9 | aggregate | plan file not found: results/exp_c/plans/T__K160__a0.90.npy |
| 160 | 0.9 | heldout_like_c | plan file not found: results/exp_c/plans/T__K160__a0.90.npy |
| 200 | 0.0 | aggregate | plan file not found: results/exp_c/plans/T__K200__a0.00.npy |
| 200 | 0.0 | heldout_like_c | plan file not found: results/exp_c/plans/T__K200__a0.00.npy |
| 200 | 0.3 | aggregate | plan file not found: results/exp_c/plans/T__K200__a0.30.npy |
| 200 | 0.3 | heldout_like_c | plan file not found: results/exp_c/plans/T__K200__a0.30.npy |
| 200 | 0.5 | aggregate | plan file not found: results/exp_c/plans/T__K200__a0.50.npy |
| 200 | 0.5 | heldout_like_c | plan file not found: results/exp_c/plans/T__K200__a0.50.npy |
| 200 | 0.7 | aggregate | plan file not found: results/exp_c/plans/T__K200__a0.70.npy |
| 200 | 0.7 | heldout_like_c | plan file not found: results/exp_c/plans/T__K200__a0.70.npy |
| 200 | 0.9 | aggregate | plan file not found: results/exp_c/plans/T__K200__a0.90.npy |
| 200 | 0.9 | heldout_like_c | plan file not found: results/exp_c/plans/T__K200__a0.90.npy |
| 300 | 0.0 | aggregate | plan file not found: results/exp_c/plans/T__K300__a0.00.npy |
| 300 | 0.0 | heldout_like_c | plan file not found: results/exp_c/plans/T__K300__a0.00.npy |
| 300 | 0.3 | aggregate | plan file not found: results/exp_c/plans/T__K300__a0.30.npy |
| 300 | 0.3 | heldout_like_c | plan file not found: results/exp_c/plans/T__K300__a0.30.npy |
| 300 | 0.5 | aggregate | plan file not found: results/exp_c/plans/T__K300__a0.50.npy |
| 300 | 0.5 | heldout_like_c | plan file not found: results/exp_c/plans/T__K300__a0.50.npy |
| 300 | 0.7 | heldout_like_c | heldout subset index not found: results/exp_c/heldout_compare_idx.npy |
| 300 | 0.9 | aggregate | plan file not found: results/exp_c/plans/T__K300__a0.90.npy |
| 300 | 0.9 | heldout_like_c | plan file not found: results/exp_c/plans/T__K300__a0.90.npy |
| 400 | 0.0 | aggregate | plan file not found: results/exp_c/plans/T__K400__a0.00.npy |
| 400 | 0.0 | heldout_like_c | plan file not found: results/exp_c/plans/T__K400__a0.00.npy |
| 400 | 0.3 | aggregate | plan file not found: results/exp_c/plans/T__K400__a0.30.npy |
| 400 | 0.3 | heldout_like_c | plan file not found: results/exp_c/plans/T__K400__a0.30.npy |
| 400 | 0.5 | aggregate | plan file not found: results/exp_c/plans/T__K400__a0.50.npy |
| 400 | 0.5 | heldout_like_c | plan file not found: results/exp_c/plans/T__K400__a0.50.npy |
| 400 | 0.7 | aggregate | plan file not found: results/exp_c/plans/T__K400__a0.70.npy |
| 400 | 0.7 | heldout_like_c | plan file not found: results/exp_c/plans/T__K400__a0.70.npy |
| 400 | 0.9 | aggregate | plan file not found: results/exp_c/plans/T__K400__a0.90.npy |
| 400 | 0.9 | heldout_like_c | plan file not found: results/exp_c/plans/T__K400__a0.90.npy |

### Random baseline

| K | alpha | scope | reason |
|---|---:|---|---|
| 0 | nan | heldout_like_c | heldout subset index not found: results/exp_random/heldout_compare_idx.npy |

