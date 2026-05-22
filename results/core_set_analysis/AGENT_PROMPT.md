# Agent prompt: fill the extended core-set metrics table

> Paste this whole file (or the "PROMPT BLOCK" section below) into the
> agent. The agent's job is to re-run a few experiment scripts so that
> the missing cells in
> `results/core_set_analysis/extended_table.md` get filled. It does
> NOT need to write new analysis code — the metric + table builder
> are already in place.

---

## What was done so far (context for the agent)

A new evaluation metric — **adaptive μ+σ core-set hit rate** — was
added to the av-bridge codebase. Implementation, table builder, and
patches to `code/run_experiments.py` are already in this branch.
See `results/core_set_analysis/REPORT.md` for the full method
description and the canonical results that have been verified.

The table at `results/core_set_analysis/extended_table.md` has 92
rows (one per (method, K, α, scope) cell). On the machine where the
metric scripts were written, only 6 of those cells could be filled
because the per-(K, α) transport plans and the heldout-row indices
are not in the cloned repo — they have to be regenerated from the
image / audio / caption embeddings.

The colleague's machine has those embeddings. Re-running the patched
experiments produces every missing artifact in one pass.

## Files for the agent to read first

| File | Why |
|---|---|
| `results/core_set_analysis/REPORT.md` | Full context: metric definition, walkthrough, why we care |
| `results/core_set_analysis/MISSING.md` | Explicit list of every cell still to be filled and the exact path each one expects |
| `code/run_experiments.py` (just glance) | See where the patched saves were added inside `exp_c_transitive`, `exp_d_caption`, `exp_unsupervised_gw`, `exp_text_only`, `exp_random`. No further code changes are required — the patches are already applied. |
| `code/core_set_metric.py` | The metric module. Skimmable; no edits expected. |
| `code/extended_metrics_table.py` | The table builder. Run once at the end. |

## What the agent must do

1. **Make sure the embeddings are on disk** under the paths the
   `load_embedding(...)` helper expects (`embeddings/vision_*.npy`,
   `embeddings/audio_*.npy`, `embeddings/ZV_text.npy`,
   `embeddings/ZA_text.npy`).
2. **Run the five patched experiments**, default encoder pair only
   (= clip-large + clap-unfused). Each command writes its sweep CSV
   AND now also saves per-(K, α) plans and the heldout-row index.

   ```bash
   python code/run_experiments.py --experiment c
   python code/run_experiments.py --experiment d
   python code/run_experiments.py --experiment unsup
   python code/run_experiments.py --experiment text
   python code/run_experiments.py --experiment random
   ```

3. **Rebuild the extended table:**

   ```bash
   python code/extended_metrics_table.py
   ```

4. **Sanity-check** that every cell is now filled:

   - Open `results/core_set_analysis/extended_table.md` — there
     should be no `—` in the rightmost ("core-hit / |core|") column
     for any OT-plan row (GW, FGW direct, FGW transitive, Random).
     Text-baseline rows continue to show the `... / 10 (R@10)`
     format by design — that's not the core-set metric.
   - Open `results/core_set_analysis/MISSING.md` — the body should
     say *"None. All core-set cells filled."*
   - The terminal output of `extended_metrics_table.py` should print
     `92/92 cells filled  (0 missing)` (the exact total may vary
     slightly if the sweep CSVs grew — what matters is that the
     "missing" count is 0).

5. **Commit + push** (on the same branch the PR is on, `fgw`):

   ```bash
   git add results/core_set_analysis/ results/exp_c/plans/ results/exp_c/heldout_compare_idx.npy \
           results/exp_d/plans/ results/exp_d/heldout_compare_idx.npy \
           results/exp_unsup/heldout_compare_idx.npy \
           results/exp_text/heldout_compare_idx.npy \
           results/exp_random/heldout_compare_idx.npy
   git commit -m "fill extended core-set metrics table with full (K, α, scope) grid"
   git push
   ```

## What the agent must NOT do

- **Do not edit the metric or table-builder scripts.** They were
  designed to consume exactly the artifact layout produced by the
  patched `run_experiments.py`. Touching them risks invalidating
  the reproducibility check (REPORT.md §6).
- **Do not change the existing canonical filenames.** The patches
  preserve `T_transitive.npy`, `T_caption.npy`, etc. and add the
  per-cell saves under `plans/` alongside them. Both are needed.
- **Do not change the seeds, encoder defaults, or K / α grids.**
  The whole point of the re-run is that every cell of the existing
  sweep CSV gets a matching plan and the metric numbers are
  reproducible.

## Expected outputs after the agent's run

```
results/
├── core_set_analysis/
│   ├── REPORT.md                    (unchanged)
│   ├── AGENT_PROMPT.md              (this file — unchanged)
│   ├── extended_table.csv           (regenerated, all cells filled)
│   ├── extended_table.json          (regenerated, all cells filled)
│   ├── extended_table.md            (regenerated, all cells filled)
│   └── MISSING.md                   (regenerated, says "None.")
├── exp_c/
│   ├── T_transitive.npy             (unchanged canonical save)
│   ├── heldout_compare_idx.npy      (NEW)
│   ├── plans/
│   │   ├── T__K10__a0.00.npy        (NEW, 40 files total)
│   │   ├── T__K10__a0.30.npy
│   │   ├── ...
│   │   └── T__K400__a0.90.npy
│   └── sweep_transitive.csv         (unchanged)
├── exp_d/
│   ├── T_caption.npy                (unchanged)
│   ├── heldout_compare_idx.npy      (NEW)
│   ├── plans/
│   │   ├── T__a0.00.npy             (NEW, 5 files)
│   │   ├── T__a0.30.npy
│   │   ├── ...
│   │   └── T__a0.90.npy
│   └── sweep.csv                    (unchanged)
├── exp_unsup/
│   ├── T_gw.npy                     (unchanged)
│   ├── heldout_compare_idx.npy      (NEW)
│   └── sweep.csv                    (unchanged)
├── exp_text/
│   ├── T_text.npy                   (unchanged)
│   ├── heldout_compare_idx.npy      (NEW)
│   └── sweep.csv                    (unchanged)
└── exp_random/
    ├── T_random.npy                 (unchanged)
    ├── heldout_compare_idx.npy      (NEW)
    └── sweep.csv                    (unchanged)
```

## Sanity values the colleague's run must reproduce

These six cells were filled and verified on the originating machine
from the canonical Ts that were already in the repo. The colleague's
re-run must reproduce them within ±0.005 (R@10 / core_hit) and
±0.5 (mean |core|):

| Method | K | α | scope | R@10 | core-hit / \|core\| |
|---|---:|---:|---|---:|---:|
| GW (unsup) | 0 | 1.0 | aggregate | 0.020 | 0.168 / 54.2 |
| FGW direct | 0 | 0.7 | aggregate | 0.225 | 0.190 / 8.8 |
| FGW transitive | 300 | 0.7 | aggregate | 0.477 | 0.568 / 18.0 |
| Random | 0 | nan | aggregate | 0.025 | 0.210 / 84.2 |
| Text | 0 | nan | aggregate | 0.233 | 0.233 / 10 (R@10) |
| Text | 0 | nan | heldout_like_c | 0.300 | 0.300 / 10 (R@10) |

If any of these six diverge after the colleague's re-run, something
is wrong — likely a different embeddings cache or a non-default
encoder being picked up. Stop and investigate before pushing.

---

## PROMPT BLOCK (copy this part if pasting into an agent)

> You are working in the cloned av-bridge repo on branch `fgw`. A
> new metric (adaptive μ+σ core-set hit rate) was added in commit
> `991c535`. The metric module, table builder, and patches to
> `code/run_experiments.py` are already in place. The only thing
> missing is the artifacts the metric needs — they were not in the
> cloned repo because they require the image / audio / caption
> embeddings.
>
> Read `results/core_set_analysis/REPORT.md` for context and
> `results/core_set_analysis/MISSING.md` for the list of cells to
> fill.
>
> Then run, in order:
>
> 1. `python code/run_experiments.py --experiment c`
> 2. `python code/run_experiments.py --experiment d`
> 3. `python code/run_experiments.py --experiment unsup`
> 4. `python code/run_experiments.py --experiment text`
> 5. `python code/run_experiments.py --experiment random`
> 6. `python code/extended_metrics_table.py`
>
> After step 6, `results/core_set_analysis/extended_table.md` should
> have no `—` placeholders in the rightmost column for any OT-plan
> row, and `MISSING.md` should report "None. All core-set cells
> filled." The terminal should print `92/92 cells filled`.
>
> The six canonical-cell numbers listed in
> `results/core_set_analysis/AGENT_PROMPT.md` ("Sanity values"
> section) must reproduce within ±0.005. If they don't, stop and
> ask the user before continuing.
>
> Do not edit `code/core_set_metric.py` or
> `code/extended_metrics_table.py`. Do not change seeds, encoder
> defaults, or K / α grids in `code/run_experiments.py`.
>
> When everything is filled and verified, commit and push.
