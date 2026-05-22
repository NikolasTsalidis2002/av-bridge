# CDF metric (core-set hit rate) across every experiment

Each cell reports `core_hit / |core|`: the fraction of rows in which the ground-truth target lies above the per-row `mu + sigma` mass threshold, and the mean size of that high-mass head. Higher hit and smaller `|core|` together indicate a sharp, correct plan.

## Random baseline

| K | alpha | scope | core_hit | |core| |
|---:|---:|---|---:|---:|
| 0 | nan | aggregate | 0.210 | 84.2 |
| 0 | nan | heldout_like_c | 0.233 | 84.6 |

## GW (unsup, alpha=1.0)

| K | alpha | scope | core_hit | |core| |
|---:|---:|---|---:|---:|
| 0 | 1.0 | aggregate | 0.168 | 54.2 |
| 0 | 1.0 | heldout_like_c | 0.300 | 53.8 |

## FGW direct (M = caption cos)

| K | alpha | scope | core_hit | |core| |
|---:|---:|---|---:|---:|
| 0 | 0.0 | aggregate | 0.092 | 2.5 |
| 0 | 0.3 | aggregate | 0.142 | 4.6 |
| 0 | 0.5 | aggregate | 0.155 | 5.7 |
| 0 | 0.7 | aggregate | 0.190 | 8.8 |
| 0 | 0.7 | heldout_like_c | 0.233 | 9.5 |
| 0 | 0.9 | aggregate | 0.292 | 22.7 |

## FGW transitive (identity bridge)

| K | alpha | scope | core_hit | |core| |
|---:|---:|---|---:|---:|
| 10 | 0.0 | aggregate | 0.152 | 40.7 |
| 10 | 0.0 | heldout_like_c | 0.133 | 44.3 |
| 10 | 0.3 | aggregate | 0.215 | 46.3 |
| 10 | 0.3 | heldout_like_c | 0.133 | 45.9 |
| 10 | 0.5 | aggregate | 0.253 | 50.7 |
| 10 | 0.5 | heldout_like_c | 0.200 | 50.3 |
| 10 | 0.7 | aggregate | 0.292 | 57.5 |
| 10 | 0.7 | heldout_like_c | 0.233 | 57.6 |
| 10 | 0.9 | aggregate | 0.315 | 66.0 |
| 10 | 0.9 | heldout_like_c | 0.167 | 64.5 |
| 20 | 0.0 | aggregate | 0.152 | 28.5 |
| 20 | 0.0 | heldout_like_c | 0.133 | 29.7 |
| 20 | 0.3 | aggregate | 0.205 | 35.5 |
| 20 | 0.3 | heldout_like_c | 0.133 | 34.8 |
| 20 | 0.5 | aggregate | 0.275 | 41.8 |
| 20 | 0.5 | heldout_like_c | 0.233 | 40.0 |
| 20 | 0.7 | aggregate | 0.300 | 50.8 |
| 20 | 0.7 | heldout_like_c | 0.333 | 49.5 |
| 20 | 0.9 | aggregate | 0.350 | 63.6 |
| 20 | 0.9 | heldout_like_c | 0.233 | 62.3 |
| 50 | 0.0 | aggregate | 0.147 | 18.3 |
| 50 | 0.0 | heldout_like_c | 0.033 | 19.9 |
| 50 | 0.3 | aggregate | 0.195 | 23.8 |
| 50 | 0.3 | heldout_like_c | 0.033 | 23.4 |
| 50 | 0.5 | aggregate | 0.225 | 29.8 |
| 50 | 0.5 | heldout_like_c | 0.033 | 28.8 |
| 50 | 0.7 | aggregate | 0.302 | 39.6 |
| 50 | 0.7 | heldout_like_c | 0.100 | 38.5 |
| 50 | 0.9 | aggregate | 0.375 | 59.8 |
| 50 | 0.9 | heldout_like_c | 0.267 | 61.4 |
| 100 | 0.0 | aggregate | 0.223 | 11.6 |
| 100 | 0.0 | heldout_like_c | 0.067 | 14.5 |
| 100 | 0.3 | aggregate | 0.273 | 15.5 |
| 100 | 0.3 | heldout_like_c | 0.100 | 20.5 |
| 100 | 0.5 | aggregate | 0.315 | 20.3 |
| 100 | 0.5 | heldout_like_c | 0.167 | 26.4 |
| 100 | 0.7 | aggregate | 0.393 | 29.5 |
| 100 | 0.7 | heldout_like_c | 0.267 | 33.7 |
| 100 | 0.9 | aggregate | 0.438 | 51.1 |
| 100 | 0.9 | heldout_like_c | 0.367 | 55.6 |
| 160 | 0.0 | aggregate | 0.330 | 8.8 |
| 160 | 0.0 | heldout_like_c | 0.067 | 9.9 |
| 160 | 0.3 | aggregate | 0.340 | 11.8 |
| 160 | 0.3 | heldout_like_c | 0.133 | 13.9 |
| 160 | 0.5 | aggregate | 0.370 | 15.4 |
| 160 | 0.5 | heldout_like_c | 0.200 | 17.4 |
| 160 | 0.7 | aggregate | 0.415 | 23.7 |
| 160 | 0.7 | heldout_like_c | 0.267 | 27.8 |
| 160 | 0.9 | aggregate | 0.448 | 46.1 |
| 160 | 0.9 | heldout_like_c | 0.400 | 51.5 |
| 200 | 0.0 | aggregate | 0.380 | 7.9 |
| 200 | 0.0 | heldout_like_c | 0.067 | 9.6 |
| 200 | 0.3 | aggregate | 0.417 | 10.5 |
| 200 | 0.3 | heldout_like_c | 0.167 | 12.9 |
| 200 | 0.5 | aggregate | 0.458 | 14.3 |
| 200 | 0.5 | heldout_like_c | 0.267 | 16.7 |
| 200 | 0.7 | aggregate | 0.460 | 21.9 |
| 200 | 0.7 | heldout_like_c | 0.300 | 26.1 |
| 200 | 0.9 | aggregate | 0.450 | 44.6 |
| 200 | 0.9 | heldout_like_c | 0.433 | 48.8 |
| 300 | 0.0 | aggregate | 0.545 | 6.1 |
| 300 | 0.0 | heldout_like_c | 0.200 | 8.6 |
| 300 | 0.3 | aggregate | 0.568 | 8.5 |
| 300 | 0.3 | heldout_like_c | 0.233 | 11.9 |
| 300 | 0.5 | aggregate | 0.570 | 11.5 |
| 300 | 0.5 | heldout_like_c | 0.300 | 15.6 |
| 300 | 0.7 | aggregate | 0.568 | 18.0 |
| 300 | 0.7 | heldout_like_c | 0.400 | 22.9 |
| 300 | 0.9 | aggregate | 0.530 | 40.2 |
| 300 | 0.9 | heldout_like_c | 0.400 | 45.7 |

