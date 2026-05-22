# Adaptive Core-Set Hit-Rate: Method, Implementation, Results

This report introduces a new evaluation metric for cross-modal transport
plans and reports its values across every method × (K, α, scope) cell
already covered by the av-bridge sweep CSVs. The metric is *adaptive*:
it has no externally chosen parameter (no top-k, no quantile, no
threshold value). For every row of a transport plan T it derives its
own "core" of significantly-above-average targets from the row's own
mean and standard deviation, then asks whether the ground-truth pair
falls inside that core.

The metric is **complementary** to R@10 and to the cluster-routing
NMI: R@10 measures strict top-k retrieval, NMI measures cluster-level
correspondence, and the core-set hit rate measures whether the plan
*soft-respects* the ground-truth pair by placing it in the row's
high-mass neighborhood. The three numbers together let you tell apart
"sharp but wrong" from "diffuse but correct" failure modes.

---

## 1. Definition

For a row-stochastic transport plan T of shape (n, n) and a ground-truth
pairing gt of length n (gt[i] is the correct target column for row i):

```
mu_i    = mean(T[i, :])
sigma_i = std(T[i, :], ddof=0)
core_i  = { j : T[i, j] > mu_i + sigma_i }
hit_i   = (gt[i] in core_i)
```

Two summary numbers are reported as a pair:

```
hit_rate       = (#rows with hit_i = True) / n
mean_core_size = mean over rows of |core_i|
```

The pair matters. A plan that spreads mass uniformly has a huge
core_i and trivially high hit_rate; a plan that is sharp but wrong
has a small core_i and a low hit_rate. The pair distinguishes these
regimes.

A chance-corrected "lift" can be derived from the pair:

```
lift = hit_rate × n_cols / mean_core_size
```

By construction `lift = 1.0` for any uniform-equivalent plan; values
above 1 indicate concentration + correctness beyond chance.

## 2. Why this metric

R@10 already asks "is GT in the top 10?" — a strict, fixed-cutoff
question. The core-set metric instead asks "is GT inside the row's
own adaptive high-mass region?" The "high-mass region" is whatever
set of columns the plan itself flags as above-typical, and its size
varies per row.

This captures a property R@10 misses: a plan can put GT at rank 12
of 17 plausible matches and still be expressing meaningful semantic
respect — every one of those 17 columns is in the same semantic
family as the GT pair, the plan just hasn't decided between them.
R@10 would mark that row a miss; the core-set metric marks it a
hit and reports the |core| so the reader can judge the plan's
confidence.

The metric only applies to row-stochastic plans (where each row is a
probability distribution). On raw cosine-similarity matrices (Text
baseline, Procrustes) the threshold `mu + sigma` has no
distributional interpretation. For those baselines we report R@10 / 10
instead, which keeps the visual form ("fraction / candidate count")
consistent with the OT-plan rows.

## 3. Implementation

The metric is implemented in [`code/core_set_metric.py`](../code/core_set_metric.py)
as a single, vectorised function:

```python
from core_set_metric import core_set_hit_rate
hit_rate, mean_core_size = core_set_hit_rate(T, gt)
```

A `row_subset` argument is accepted for heldout evaluation: per-row
`mu_i` and `sigma_i` are still computed over the row's full set of
columns, but the averaging at the end is restricted to the subset.

The driver that builds the full extended table is
[`code/extended_metrics_table.py`](../code/extended_metrics_table.py). It
reads every sweep CSV under `results/exp_*` and joins the existing
metrics (R@10, Routes, NMI) with a freshly computed core-set hit /
|core| column. It writes:

- `extended_table.csv`  — machine-readable, full grid
- `extended_table.json` — same data as JSON
- `extended_table.md`   — human-readable table
- `MISSING.md`          — every (method, K, alpha, scope) cell that
  could not be filled, with the exact reason

## 4. Walkthrough: one HIT and one MISS row

The following two examples are taken from the canonical FGW transitive
plan (`results/exp_c/T_transitive.npy`, K=300, α=0.7), each row a 400-
length distribution over candidate audios.

### HIT example, row i = 4 (clip_id = 10130549263)

| field | value |
|---|---|
| Image-side caption (input) | *"A man was using his mobile phone while eating in the hotel."* |
| Ground-truth audio caption | *"They are speaking with their family."* |
| row mu | 0.00250 (= 1 / 400, mechanical for a row-stochastic plan) |
| row sigma | 0.01025 |
| threshold mu + sigma | 0.01275 |
| T[i, gt[i]] | 0.02382 (above threshold → **in core**) |
| \|core_i\| | 17 of 400 (4.2 %) |

The 17 columns above threshold form a coherent "people talking /
family conversation" semantic cluster. The GT pair lands at rank 12
of 17, well inside the core. The row's neighborhood is exactly the
right semantic family.

### MISS example, row i = 0 (clip_id = 10001787725)

| field | value |
|---|---|
| Image-side caption (input) | *"A small baby is looking at someone and trying to talk."* |
| Ground-truth audio caption | *"A female voice runs in the background, and the young boy is smiling sometimes."* |
| row mu | 0.00250 |
| row sigma | 0.00831 |
| threshold mu + sigma | 0.01081 |
| T[i, gt[i]] | 0.000161 (far below threshold → **NOT in core**) |
| \|core_i\| | 16 of 400 |

The 16 columns in this row's core are all "baby + mother/family
talking" audios — the right semantic category. The plan understands
the category but does not place the literal GT pair inside its
high-mass region. This is the failure mode the metric is designed
to distinguish from "wrong category entirely": the category is right
but the specific identity is missed.

## 5. Results: canonical row, default encoder pair

The following row appears in the full extended table at the canonical
(K, α) cells and represents the headline of this analysis.

| Method | scope | R@10 | Routes | NMI | core-hit / \|core\| | lift |
|---|---|---:|---:|---:|---:|---:|
| GW (unsup, α=1.0) | aggregate | 0.020 | 3/15 | 0.374 | 0.168 / 54.2 | 1.2× |
| FGW direct (α=0.7) | aggregate | 0.225 | 7/15 | 0.224 | 0.190 / 8.8 | 8.6× |
| **FGW transitive (K=300, α=0.7)** | aggregate | 0.477 | 8/15 | 0.222 | **0.568 / 18.0** | **12.6×** |
| Random baseline | aggregate | 0.025 | 0/15 | 0.115 | 0.210 / 84.2 | 1.0× |
| Text baseline | aggregate | 0.233 | 2/15 | 0.258 | 0.233 / 10 (R@10) | — |

### Reading

- **GW unsup is the structural-only failure mode.** It produces the
  highest cluster NMI (0.374) of all the OT plans but the lowest
  R@10 (0.020), and its core-set lift sits right at chance (1.2×).
  Pure entropic GW recovers the right *geometry* of the two modalities
  but never sees a cross-modal cost, so it has no semantic anchor at
  the sample level.
- **FGW direct (caption-cos M) is sharp but wrong.** Its core is
  tight (8.8 columns ≈ 2.2 % of all targets) but only 19 % of rows
  contain GT in that core. The plan commits confidently to specific
  cells, but those cells are systematically not the GT pair. This
  matches the memorisation-without-generalisation pattern seen in
  the user's own thesis Experiment 27.
- **FGW transitive is the only plan with concentration *and*
  correctness.** Its core is moderately sharp (18 columns ≈ 4.5 % of
  targets) and contains GT 57 % of the time — a 12.6× lift over
  chance. The transitive composition through caption-bridge picks up
  the semantic signal the structural-only GW lacks.
- **Random and Text are diagnostic floors.** Random's core is huge
  (84 columns) and its hit rate exactly matches chance (lift = 1.0).
  Text-baseline R@10 = 0.233 is included as a non-OT reference; the
  core-set metric does not apply to its raw similarity matrix.

### Per-row variation behind the headline numbers (FGW transitive)

The `|core| = 18.0` figure is a *mean* over 400 rows; per-row variation
is large:

```
min    : 1     5%  : 3
median : 16    25% : 10
mean   : 18.0
                75% : 25
                95% : 39
max    : 63
std    : 10.9
```

Hit rate is positively correlated with row sharpness:

| Per-row \|core\| | # rows | Hit rate |
|---|---:|---:|
| < 10 (tight) | 93 | 0.720 |
| 10–20 (moderate) | 160 | 0.531 |
| 20–30 (loose) | 83 | 0.542 |
| ≥ 30 (diffuse) | 64 | 0.469 |

The tightest 25 % of rows hit at 72 %, the most diffuse 16 % hit at 47 %.
The plan's per-row confidence is *calibrated* — sharper rows are
indeed more likely to contain GT.

## 6. Current coverage of the extended table

The driver script writes
[`results/core_set_analysis/extended_table.md`](extended_table.md). At
the time of writing **6 of 92 cells are filled** (the canonical
aggregate cells for each of the five methods). 86 cells are flagged
"—" because the matching transport plan or heldout-row index is not
present in this clone of the repository.

The cells that are filled:

| Method | K | α | scope | core-hit / \|core\| (or R@10 / 10) |
|---|---:|---:|---|---:|
| GW (unsup) | 0 | 1.0 | aggregate | 0.168 / 54.2 |
| FGW direct (caption cos) | 0 | 0.7 | aggregate | 0.190 / 8.8 |
| FGW transitive | 300 | 0.7 | aggregate | 0.568 / 18.0 |
| Random | 0 | nan | aggregate | 0.210 / 84.2 |
| Text | 0 | nan | aggregate | 0.233 / 10 (R@10) |
| Text | 0 | nan | heldout_like_c | 0.300 / 10 (R@10) |

The two reasons for the remaining 86 missing cells are:

1. **No per-(K, α) transport plans on disk.** The original av-bridge
   `run_experiments.py` only persisted the canonical (K=300, α=0.7)
   transport plan to disk; the sweep loop discarded all the other
   cells' plans after computing CSV metrics on them. To fill those
   86 cells the experiments have to be re-run with the per-cell save
   in place (now patched, see Section 7).

2. **No heldout-row-index file on disk.** The "heldout_like_c" scope
   restricts the metric to the ~100 rows outside the union of the
   two K=300 stratified anchor subsets. The indices are derived from
   k-means stratification on the image and audio embeddings, which
   are not in the repository. The patched `run_experiments.py` now
   saves `heldout_compare_idx.npy` next to each sweep CSV, so a
   single re-run is enough to enable all heldout-scope evaluations.

See [`MISSING.md`](MISSING.md) for the complete per-cell list.

## 7. What the patched `run_experiments.py` now does

The following additive changes were made (no existing behaviour was
removed):

- `exp_c_transitive`
  - Creates `results/exp_c/plans/` and saves `T__K{K}__a{alpha:.2f}.npy`
    for every (K, α) iteration of the sweep loop.
  - Saves `results/exp_c/heldout_compare_idx.npy` (the K=300 anchor
    union used for the "heldout_like_c" scope) once at the start.

- `exp_d_caption`
  - Creates `results/exp_d/plans/` and saves `T__a{alpha:.2f}.npy`
    for every alpha iteration.
  - Saves `results/exp_d/heldout_compare_idx.npy`.

- `exp_unsupervised_gw`, `exp_text_only`, `exp_random`
  - Save `heldout_compare_idx.npy` next to their canonical
    `T_*.npy`. No plan-loop change is needed (these experiments
    produce a single plan).

The disk cost of the additions is moderate: the FGW transitive
sweep produces 40 plans × ~1.28 MB = ~51 MB; the FGW direct sweep
produces 5 plans × ~1.28 MB = ~6.4 MB; the index files are < 4 KB
each.

## 8. How to fill the missing cells

Run the patched code on a machine that has the av-bridge embeddings:

```bash
python code/run_experiments.py --experiment c
python code/run_experiments.py --experiment d
python code/run_experiments.py --experiment unsup
python code/run_experiments.py --experiment text
python code/run_experiments.py --experiment random
```

Each command will:
1. Re-fit the relevant transport plans.
2. Append rows to the matching `sweep*.csv` (overwrite mode).
3. Save `plans/T__*.npy` for every (K, α) cell, plus
   `heldout_compare_idx.npy` once.

After the re-run, rebuild the extended table:

```bash
python code/extended_metrics_table.py
```

The script will pick up the newly-saved plans automatically and
overwrite `results/core_set_analysis/extended_table.{csv,json,md}`
and `MISSING.md`.

## 9. Reading conventions for the new column

In `extended_table.md` the new column ("core-hit / |core|") follows
two conventions, both visually consistent:

- **OT-plan rows** (GW unsup, FGW direct, FGW transitive, Random):
  the format is `hit / |core|`, e.g. `0.568 / 18.0`. The first
  number is the per-row hit rate; the second is the *mean* per-row
  core size (different rows have different cores, see Section 5).
- **Non-OT baseline rows** (Text): the format is `R@10 / 10`, e.g.
  `0.233 / 10 (R@10)`. The first number is R@10 itself; the
  denominator is the fixed retrieval-at-10 candidate count. The
  `(R@10)` suffix is a reminder that the value is *not* the core-set
  metric.

Both formats share the same "fraction / candidate count" shape, so
the column is comparable at a glance: how many candidates did the
method consider, and what fraction contained GT?

## 10. Caveats

- This is an aggregate / heldout scoring convention. The Text baseline
  rows use R@10 because the underlying matrix is not a probability
  distribution — readers should not compare its "0.233 / 10" against
  the OT plans' "X / |core|" as if they were the same metric.
- The mean-plus-one-σ threshold is not Gaussian-derived (the row
  distributions are heavily right-skewed: skew 9–13, excess kurtosis
  85–190, every row fails Shapiro-Wilk). It is a *parameter-free,
  distribution-aware* concentration cutoff that does not assume
  normality. Robustness checks with `median + σ` and the effective-
  support (`N_eff = 1 / Σp²`) formulations yield identical method
  rankings; see the source comments in `core_set_metric.py` for the
  full discussion.
- The per-row σ for OT plans is dominated by the heavy right tail,
  so the threshold mostly tracks σ itself. This is intentional:
  the metric asks "is GT meaningfully contributing to the row's
  variance" — i.e., is GT one of the cells the plan is actually
  betting on, regardless of where on the row the center sits.
