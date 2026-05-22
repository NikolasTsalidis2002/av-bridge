# Recall@threshold (μ+σ head) across every experiment

## What each column means (in plain words)

- **K** — number of paired anchors used during training (per modality).
  Only the transitive bridge varies this; for the others K is fixed or
  N/A (no training).
- **α** — FGW blend. `α = 0` means the plan is driven purely by the
  caption-cost feature term; `α = 1` means it is driven purely by the
  GW structural term. Everything in between mixes the two.
- **scope** — `aggregate` is all 400 queries; `heldout_like_c` is the
  ~30 queries that were *not* used as training anchors at the
  reusable budget. The held-out number is the honest test reading
  for the supervised recipes.
- **samples_to_threshold** — on average per query, how many of the
  400 target audios receive enough transport mass to clear that
  row's `μ + σ` line. In plain words: **how many "winners" each
  image has, where a winner is a column whose mass is more than
  one standard deviation above the row's average mass.** A small
  number = sharp plan; a large number = diffuse plan.
- **recall_at_threshold** — out of all queries in the scope, the
  fraction whose correct (ground-truth) audio is one of those
  winners. In plain words: **how often the right answer is among
  the winners.** This is `R@K` with `K = samples_to_threshold`
  chosen per row by the μ+σ rule.

The metric only makes sense on row-stochastic plans (transport mass
is a real probability distribution per row). The text-only baseline
is omitted because its matrix is raw cosine similarity, so the
μ+σ rule has no probabilistic meaning there.

---

## Random baseline

| K | α | scope          | samples_to_threshold | recall_at_threshold |
|---:|---:|---|---:|---:|
| 0 | — | aggregate      | 84.2 | 0.210 |
| 0 | — | heldout_like_c | 84.6 | 0.233 |

## GW (Pure-GW, α=1.0)

| K | α | scope          | samples_to_threshold | recall_at_threshold |
|---:|---:|---|---:|---:|
| 0 | 1.0 | aggregate      | 54.2 | 0.168 |
| 0 | 1.0 | heldout_like_c | 53.8 | 0.300 |

## FGW direct (M = caption cosine)

| K | α | scope          | samples_to_threshold | recall_at_threshold |
|---:|---:|---|---:|---:|
| 0 | 0.0 | aggregate      |  2.5 | 0.092 |
| 0 | 0.3 | aggregate      |  4.6 | 0.142 |
| 0 | 0.5 | aggregate      |  5.7 | 0.155 |
| 0 | 0.7 | aggregate      |  8.8 | 0.190 |
| 0 | 0.7 | heldout_like_c |  9.5 | 0.233 |
| 0 | 0.9 | aggregate      | 22.7 | 0.292 |

## FGW transitive (identity bridge) — aggregate

| K | α | samples_to_threshold | recall_at_threshold |
|---:|---:|---:|---:|
|  10 | 0.0 | 40.7 | 0.152 |
|  10 | 0.3 | 46.3 | 0.215 |
|  10 | 0.5 | 50.7 | 0.253 |
|  10 | 0.7 | 57.5 | 0.292 |
|  10 | 0.9 | 66.0 | 0.315 |
|  20 | 0.0 | 28.5 | 0.152 |
|  20 | 0.3 | 35.5 | 0.205 |
|  20 | 0.5 | 41.8 | 0.275 |
|  20 | 0.7 | 50.8 | 0.300 |
|  20 | 0.9 | 63.6 | 0.350 |
|  50 | 0.0 | 18.3 | 0.147 |
|  50 | 0.3 | 23.8 | 0.195 |
|  50 | 0.5 | 29.8 | 0.225 |
|  50 | 0.7 | 39.6 | 0.302 |
|  50 | 0.9 | 59.8 | 0.375 |
| 100 | 0.0 | 11.6 | 0.223 |
| 100 | 0.3 | 15.5 | 0.273 |
| 100 | 0.5 | 20.3 | 0.315 |
| 100 | 0.7 | 29.5 | 0.393 |
| 100 | 0.9 | 51.1 | 0.438 |
| 160 | 0.0 |  8.8 | 0.330 |
| 160 | 0.3 | 11.8 | 0.340 |
| 160 | 0.5 | 15.4 | 0.370 |
| 160 | 0.7 | 23.7 | 0.415 |
| 160 | 0.9 | 46.1 | 0.448 |
| 200 | 0.0 |  7.9 | 0.380 |
| 200 | 0.3 | 10.5 | 0.417 |
| 200 | 0.5 | 14.3 | 0.458 |
| 200 | 0.7 | 21.9 | 0.460 |
| 200 | 0.9 | 44.6 | 0.450 |
| 300 | 0.0 |  6.1 | 0.545 |
| 300 | 0.3 |  8.5 | 0.568 |
| 300 | 0.5 | 11.5 | 0.570 |
| 300 | 0.7 | 18.0 | 0.568 |
| 300 | 0.9 | 40.2 | 0.530 |

## FGW transitive (identity bridge) — heldout_like_c

| K | α | samples_to_threshold | recall_at_threshold |
|---:|---:|---:|---:|
|  10 | 0.0 | 44.3 | 0.133 |
|  10 | 0.3 | 45.9 | 0.133 |
|  10 | 0.5 | 50.3 | 0.200 |
|  10 | 0.7 | 57.6 | 0.233 |
|  10 | 0.9 | 64.5 | 0.167 |
|  20 | 0.0 | 29.7 | 0.133 |
|  20 | 0.3 | 34.8 | 0.133 |
|  20 | 0.5 | 40.0 | 0.233 |
|  20 | 0.7 | 49.5 | 0.333 |
|  20 | 0.9 | 62.3 | 0.233 |
|  50 | 0.0 | 19.9 | 0.033 |
|  50 | 0.3 | 23.4 | 0.033 |
|  50 | 0.5 | 28.8 | 0.033 |
|  50 | 0.7 | 38.5 | 0.100 |
|  50 | 0.9 | 61.4 | 0.267 |
| 100 | 0.0 | 14.5 | 0.067 |
| 100 | 0.3 | 20.5 | 0.100 |
| 100 | 0.5 | 26.4 | 0.167 |
| 100 | 0.7 | 33.7 | 0.267 |
| 100 | 0.9 | 55.6 | 0.367 |
| 160 | 0.0 |  9.9 | 0.067 |
| 160 | 0.3 | 13.9 | 0.133 |
| 160 | 0.5 | 17.4 | 0.200 |
| 160 | 0.7 | 27.8 | 0.267 |
| 160 | 0.9 | 51.5 | 0.400 |
| 200 | 0.0 |  9.6 | 0.067 |
| 200 | 0.3 | 12.9 | 0.167 |
| 200 | 0.5 | 16.7 | 0.267 |
| 200 | 0.7 | 26.1 | 0.300 |
| 200 | 0.9 | 48.8 | 0.433 |
| 300 | 0.0 |  8.6 | 0.200 |
| 300 | 0.3 | 11.9 | 0.233 |
| 300 | 0.5 | 15.6 | 0.300 |
| 300 | 0.7 | 22.9 | 0.400 |
| 300 | 0.9 | 45.7 | 0.400 |

## Text baseline (raw caption cosine)

Excluded — `T_text.npy` is a raw similarity matrix, not row-stochastic.
The μ+σ rule has no probabilistic interpretation there. The thesis
reports R@10 (0.233 aggregate / 0.300 held-out) as the appropriate
substitute for this baseline.
