# Cumulative nucleus metric

For each row of the plan, sort target columns by mass (descending) and take the smallest set whose cumulative mass reaches *p*. **nucleus_size** is the size of that set (averaged over rows); **nucleus_hit** is the fraction of rows where the GT column lies in that set. Reported at p ∈ {0.5, 0.9}.

Text baseline is omitted because `T_text.npy` is a raw similarity matrix, not row-stochastic.

## Random baseline

| K | α | scope | p | nucleus_hit | nucleus_size |
|---:|---:|---|---:|---:|---:|
| 0 | — | aggregate | 0.5 | 0.315 | 117.9 |
| 0 | — | aggregate | 0.9 | 0.685 | 274.2 |
| 0 | — | heldout_like_c | 0.5 | 0.367 | 117.9 |
| 0 | — | heldout_like_c | 0.9 | 0.833 | 273.9 |

## GW (unsup, alpha=1.0)

| K | α | scope | p | nucleus_hit | nucleus_size |
|---:|---:|---|---:|---:|---:|
| 0 | 1.0 | aggregate | 0.5 | 0.115 | 43.9 |
| 0 | 1.0 | aggregate | 0.9 | 0.388 | 134.5 |
| 0 | 1.0 | heldout_like_c | 0.5 | 0.167 | 42.9 |
| 0 | 1.0 | heldout_like_c | 0.9 | 0.433 | 123.9 |

## FGW direct (M = caption cos)

| K | α | scope | p | nucleus_hit | nucleus_size |
|---:|---:|---|---:|---:|---:|
| 0 | 0.0 | aggregate | 0.5 | 0.152 | 38.0 |
| 0 | 0.0 | aggregate | 0.9 | 0.165 | 38.5 |
| 0 | 0.0 | heldout_like_c | 0.5 | 0.200 | 40.9 |
| 0 | 0.0 | heldout_like_c | 0.9 | 0.200 | 41.4 |
| 0 | 0.3 | aggregate | 0.5 | 0.062 | 1.6 |
| 0 | 0.3 | aggregate | 0.9 | 0.135 | 4.7 |
| 0 | 0.3 | heldout_like_c | 0.5 | 0.133 | 1.9 |
| 0 | 0.3 | heldout_like_c | 0.9 | 0.200 | 5.6 |
| 0 | 0.5 | aggregate | 0.5 | 0.080 | 2.0 |
| 0 | 0.5 | aggregate | 0.9 | 0.163 | 7.2 |
| 0 | 0.5 | heldout_like_c | 0.5 | 0.133 | 2.4 |
| 0 | 0.5 | heldout_like_c | 0.9 | 0.200 | 8.6 |
| 0 | 0.7 | aggregate | 0.5 | 0.105 | 3.2 |
| 0 | 0.7 | aggregate | 0.9 | 0.255 | 16.6 |
| 0 | 0.7 | heldout_like_c | 0.5 | 0.133 | 3.3 |
| 0 | 0.7 | heldout_like_c | 0.9 | 0.300 | 15.3 |
| 0 | 0.9 | aggregate | 0.5 | 0.200 | 15.6 |
| 0 | 0.9 | aggregate | 0.9 | 0.550 | 77.4 |
| 0 | 0.9 | heldout_like_c | 0.5 | 0.167 | 11.7 |
| 0 | 0.9 | heldout_like_c | 0.9 | 0.533 | 62.7 |

## FGW transitive (identity bridge)

| K | α | scope | p | nucleus_hit | nucleus_size |
|---:|---:|---|---:|---:|---:|
| 10 | 0.0 | aggregate | 0.5 | 0.245 | 69.3 |
| 10 | 0.0 | aggregate | 0.9 | 0.695 | 225.5 |
| 10 | 0.0 | heldout_like_c | 0.5 | 0.200 | 71.5 |
| 10 | 0.0 | heldout_like_c | 0.9 | 0.600 | 228.7 |
| 10 | 0.3 | aggregate | 0.5 | 0.323 | 75.8 |
| 10 | 0.3 | aggregate | 0.9 | 0.733 | 231.5 |
| 10 | 0.3 | heldout_like_c | 0.5 | 0.200 | 78.2 |
| 10 | 0.3 | heldout_like_c | 0.9 | 0.700 | 231.8 |
| 10 | 0.5 | aggregate | 0.5 | 0.362 | 82.8 |
| 10 | 0.5 | aggregate | 0.9 | 0.800 | 239.7 |
| 10 | 0.5 | heldout_like_c | 0.5 | 0.300 | 85.7 |
| 10 | 0.5 | heldout_like_c | 0.9 | 0.767 | 237.7 |
| 10 | 0.7 | aggregate | 0.5 | 0.407 | 93.2 |
| 10 | 0.7 | aggregate | 0.9 | 0.845 | 252.5 |
| 10 | 0.7 | heldout_like_c | 0.5 | 0.300 | 95.5 |
| 10 | 0.7 | heldout_like_c | 0.9 | 0.800 | 248.0 |
| 10 | 0.9 | aggregate | 0.5 | 0.495 | 109.1 |
| 10 | 0.9 | aggregate | 0.9 | 0.895 | 267.6 |
| 10 | 0.9 | heldout_like_c | 0.5 | 0.433 | 108.0 |
| 10 | 0.9 | heldout_like_c | 0.9 | 0.933 | 259.0 |
| 20 | 0.0 | aggregate | 0.5 | 0.130 | 31.7 |
| 20 | 0.0 | aggregate | 0.9 | 0.605 | 142.0 |
| 20 | 0.0 | heldout_like_c | 0.5 | 0.100 | 32.4 |
| 20 | 0.0 | heldout_like_c | 0.9 | 0.633 | 141.9 |
| 20 | 0.3 | aggregate | 0.5 | 0.242 | 45.6 |
| 20 | 0.3 | aggregate | 0.9 | 0.690 | 171.7 |
| 20 | 0.3 | heldout_like_c | 0.5 | 0.300 | 45.1 |
| 20 | 0.3 | heldout_like_c | 0.9 | 0.733 | 172.4 |
| 20 | 0.5 | aggregate | 0.5 | 0.300 | 58.2 |
| 20 | 0.5 | aggregate | 0.9 | 0.767 | 194.9 |
| 20 | 0.5 | heldout_like_c | 0.5 | 0.267 | 58.5 |
| 20 | 0.5 | heldout_like_c | 0.9 | 0.800 | 199.0 |
| 20 | 0.7 | aggregate | 0.5 | 0.383 | 75.7 |
| 20 | 0.7 | aggregate | 0.9 | 0.838 | 223.2 |
| 20 | 0.7 | heldout_like_c | 0.5 | 0.367 | 77.9 |
| 20 | 0.7 | heldout_like_c | 0.9 | 0.833 | 227.6 |
| 20 | 0.9 | aggregate | 0.5 | 0.465 | 101.8 |
| 20 | 0.9 | aggregate | 0.9 | 0.917 | 258.5 |
| 20 | 0.9 | heldout_like_c | 0.5 | 0.400 | 102.9 |
| 20 | 0.9 | heldout_like_c | 0.9 | 0.933 | 254.6 |
| 50 | 0.0 | aggregate | 0.5 | 0.098 | 11.9 |
| 50 | 0.0 | aggregate | 0.9 | 0.355 | 66.2 |
| 50 | 0.0 | heldout_like_c | 0.5 | 0.000 | 13.0 |
| 50 | 0.0 | heldout_like_c | 0.9 | 0.167 | 72.2 |
| 50 | 0.3 | aggregate | 0.5 | 0.145 | 20.4 |
| 50 | 0.3 | aggregate | 0.9 | 0.517 | 99.3 |
| 50 | 0.3 | heldout_like_c | 0.5 | 0.000 | 21.8 |
| 50 | 0.3 | heldout_like_c | 0.9 | 0.333 | 109.1 |
| 50 | 0.5 | aggregate | 0.5 | 0.203 | 31.3 |
| 50 | 0.5 | aggregate | 0.9 | 0.632 | 131.5 |
| 50 | 0.5 | heldout_like_c | 0.5 | 0.033 | 34.5 |
| 50 | 0.5 | heldout_like_c | 0.9 | 0.467 | 142.6 |
| 50 | 0.7 | aggregate | 0.5 | 0.318 | 51.5 |
| 50 | 0.7 | aggregate | 0.9 | 0.762 | 176.2 |
| 50 | 0.7 | heldout_like_c | 0.5 | 0.133 | 56.7 |
| 50 | 0.7 | heldout_like_c | 0.9 | 0.700 | 183.0 |
| 50 | 0.9 | aggregate | 0.5 | 0.448 | 88.4 |
| 50 | 0.9 | aggregate | 0.9 | 0.895 | 238.3 |
| 50 | 0.9 | heldout_like_c | 0.5 | 0.400 | 89.1 |
| 50 | 0.9 | heldout_like_c | 0.9 | 0.933 | 237.1 |
| 100 | 0.0 | aggregate | 0.5 | 0.125 | 5.2 |
| 100 | 0.0 | aggregate | 0.9 | 0.383 | 29.5 |
| 100 | 0.0 | heldout_like_c | 0.5 | 0.033 | 7.5 |
| 100 | 0.0 | heldout_like_c | 0.9 | 0.233 | 40.7 |
| 100 | 0.3 | aggregate | 0.5 | 0.163 | 9.1 |
| 100 | 0.3 | aggregate | 0.9 | 0.525 | 51.7 |
| 100 | 0.3 | heldout_like_c | 0.5 | 0.067 | 13.1 |
| 100 | 0.3 | heldout_like_c | 0.9 | 0.400 | 68.5 |
| 100 | 0.5 | aggregate | 0.5 | 0.217 | 15.4 |
| 100 | 0.5 | aggregate | 0.9 | 0.632 | 79.4 |
| 100 | 0.5 | heldout_like_c | 0.5 | 0.100 | 21.5 |
| 100 | 0.5 | heldout_like_c | 0.9 | 0.433 | 100.3 |
| 100 | 0.7 | aggregate | 0.5 | 0.333 | 30.8 |
| 100 | 0.7 | aggregate | 0.9 | 0.748 | 129.2 |
| 100 | 0.7 | heldout_like_c | 0.5 | 0.167 | 38.0 |
| 100 | 0.7 | heldout_like_c | 0.9 | 0.733 | 147.9 |
| 100 | 0.9 | aggregate | 0.5 | 0.512 | 74.3 |
| 100 | 0.9 | aggregate | 0.9 | 0.875 | 218.9 |
| 100 | 0.9 | heldout_like_c | 0.5 | 0.433 | 79.8 |
| 100 | 0.9 | heldout_like_c | 0.9 | 0.933 | 222.3 |
| 160 | 0.0 | aggregate | 0.5 | 0.205 | 3.3 |
| 160 | 0.0 | aggregate | 0.9 | 0.372 | 16.5 |
| 160 | 0.0 | heldout_like_c | 0.5 | 0.067 | 3.8 |
| 160 | 0.0 | heldout_like_c | 0.9 | 0.067 | 18.9 |
| 160 | 0.3 | aggregate | 0.5 | 0.217 | 5.4 |
| 160 | 0.3 | aggregate | 0.9 | 0.480 | 30.5 |
| 160 | 0.3 | heldout_like_c | 0.5 | 0.067 | 6.1 |
| 160 | 0.3 | heldout_like_c | 0.9 | 0.267 | 35.3 |
| 160 | 0.5 | aggregate | 0.5 | 0.250 | 9.1 |
| 160 | 0.5 | aggregate | 0.9 | 0.590 | 51.9 |
| 160 | 0.5 | heldout_like_c | 0.5 | 0.100 | 10.5 |
| 160 | 0.5 | heldout_like_c | 0.9 | 0.333 | 62.5 |
| 160 | 0.7 | aggregate | 0.5 | 0.338 | 20.5 |
| 160 | 0.7 | aggregate | 0.9 | 0.715 | 96.8 |
| 160 | 0.7 | heldout_like_c | 0.5 | 0.233 | 24.2 |
| 160 | 0.7 | heldout_like_c | 0.9 | 0.600 | 113.0 |
| 160 | 0.9 | aggregate | 0.5 | 0.495 | 64.3 |
| 160 | 0.9 | aggregate | 0.9 | 0.885 | 201.0 |
| 160 | 0.9 | heldout_like_c | 0.5 | 0.467 | 71.4 |
| 160 | 0.9 | heldout_like_c | 0.9 | 0.900 | 207.1 |
| 200 | 0.0 | aggregate | 0.5 | 0.258 | 2.9 |
| 200 | 0.0 | aggregate | 0.9 | 0.445 | 13.4 |
| 200 | 0.0 | heldout_like_c | 0.5 | 0.033 | 3.4 |
| 200 | 0.0 | heldout_like_c | 0.9 | 0.100 | 16.4 |
| 200 | 0.3 | aggregate | 0.5 | 0.287 | 4.5 |
| 200 | 0.3 | aggregate | 0.9 | 0.517 | 24.3 |
| 200 | 0.3 | heldout_like_c | 0.5 | 0.067 | 5.6 |
| 200 | 0.3 | heldout_like_c | 0.9 | 0.333 | 30.3 |
| 200 | 0.5 | aggregate | 0.5 | 0.330 | 7.6 |
| 200 | 0.5 | aggregate | 0.9 | 0.585 | 41.9 |
| 200 | 0.5 | heldout_like_c | 0.5 | 0.133 | 9.1 |
| 200 | 0.5 | heldout_like_c | 0.9 | 0.367 | 52.4 |
| 200 | 0.7 | aggregate | 0.5 | 0.375 | 17.0 |
| 200 | 0.7 | aggregate | 0.9 | 0.725 | 82.5 |
| 200 | 0.7 | heldout_like_c | 0.5 | 0.233 | 20.5 |
| 200 | 0.7 | heldout_like_c | 0.9 | 0.633 | 94.1 |
| 200 | 0.9 | aggregate | 0.5 | 0.505 | 58.5 |
| 200 | 0.9 | aggregate | 0.9 | 0.880 | 189.5 |
| 200 | 0.9 | heldout_like_c | 0.5 | 0.500 | 64.6 |
| 200 | 0.9 | heldout_like_c | 0.9 | 0.933 | 196.5 |
| 300 | 0.0 | aggregate | 0.5 | 0.393 | 2.2 |
| 300 | 0.0 | aggregate | 0.9 | 0.578 | 8.8 |
| 300 | 0.0 | heldout_like_c | 0.5 | 0.100 | 2.9 |
| 300 | 0.0 | heldout_like_c | 0.9 | 0.200 | 13.4 |
| 300 | 0.3 | aggregate | 0.5 | 0.395 | 3.2 |
| 300 | 0.3 | aggregate | 0.9 | 0.625 | 16.1 |
| 300 | 0.3 | heldout_like_c | 0.5 | 0.133 | 4.6 |
| 300 | 0.3 | heldout_like_c | 0.9 | 0.333 | 22.7 |
| 300 | 0.5 | aggregate | 0.5 | 0.430 | 5.2 |
| 300 | 0.5 | aggregate | 0.9 | 0.680 | 28.7 |
| 300 | 0.5 | heldout_like_c | 0.5 | 0.200 | 7.6 |
| 300 | 0.5 | heldout_like_c | 0.9 | 0.400 | 39.8 |
| 300 | 0.7 | aggregate | 0.5 | 0.470 | 12.0 |
| 300 | 0.7 | aggregate | 0.9 | 0.772 | 60.4 |
| 300 | 0.7 | heldout_like_c | 0.5 | 0.267 | 15.9 |
| 300 | 0.7 | heldout_like_c | 0.9 | 0.600 | 77.2 |
| 300 | 0.9 | aggregate | 0.5 | 0.550 | 49.3 |
| 300 | 0.9 | aggregate | 0.9 | 0.892 | 170.7 |
| 300 | 0.9 | heldout_like_c | 0.5 | 0.467 | 57.4 |
| 300 | 0.9 | heldout_like_c | 0.9 | 0.967 | 184.3 |

