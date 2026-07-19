# GPU retrain (100 epochs) — failure analysis

Real measured result: **val Dice 0.2127**, confirmed by directly re-running
evaluation against this checkpoint. That is *worse* than the existing CPU
baseline (0.4603, see `../../FAILURE_ANALYSIS.md`). Per the honesty rule this
project runs on, the baseline checkpoint (`checkpoints/best_spleen_model.pth`)
has **not** been replaced — this run is documented as a failure, not shipped.

## Confirming the number

`evaluate_spleen.py` is hardcoded for the CPU baseline's architecture
(4-level UNet, 64³ patches, 2.5/2.5/3.0mm spacing) and fails with a
`state_dict` size mismatch against this checkpoint. This run actually used
the 5-level UNet from `train_spleen_colab.py` (channels
16/32/64/128/256, 96³ patches, 1.5/1.5/2.0mm spacing). `evaluate_gpu_run.py`
in this folder is a copy of the eval script pointed at the correct
architecture/spacing for this checkpoint; running it reproduces **0.2127**
exactly, matching the "best" value logged at epoch 90 in `training_log.csv`.
The result is appended to the repo's `benchmark_log.csv`.

## What the curve actually shows

![training curve](training_curve.png)

Looking at `training_curve.png` / `training_log.csv`, there is **no early
plateau at epoch 10** — val Dice climbs fairly steadily and noisily from
0.073 (epoch 5) up to its peak of 0.2127 (epoch 90). From roughly epoch 50
onward it oscillates in a 0.15–0.21 band without a clean plateau, and the
last two logged points (epoch 95: 0.2077, epoch 100: 0.2087) sit essentially
flat relative to the epoch-90 peak — not a hard ceiling reached early, more
like a slow, still-not-fully-converged climb that was cut off.

## The real, verified cause: the run stopped at 1/3 of its planned LR schedule

`train_spleen_colab.py` configures `CosineAnnealingLR` with
`T_max=NUM_EPOCHS=300` — the whole schedule assumes 300 epochs. This run
(folder name `gpu_run_100ep`, confirmed by `training_log.csv` having exactly
100 rows) only completed 100. At epoch 100, cosine annealing has decayed the
LR to only ~75% of its peak value (1e-3 → ~7.5e-4) — the model never reached
the low-LR refinement phase the schedule was designed to deliver. Training
for a third of the intended schedule while the LR is still high is
consistent with the noisy, still-rising curve above: this looks like a run
that was interrupted mid-training (most likely a Colab session timeout,
given the script's own comment estimating a 1–2.5hr T4 runtime for the full
300 epochs), not one that converged and plateaued.

## Secondary factor: capacity + augmentation vs. data size

Independent of the truncated schedule, this run pairs a much bigger model
(5-level UNet vs. the baseline's 4-level) and heavier augmentation (random
flips on all 3 axes, `RandRotate90d`, intensity shift, Gaussian noise)
against the same 33 training cases the CPU baseline used (41 total Decathlon
training cases minus 8 held out for validation), just with a substantially
more aggressive pipeline. More capacity and heavier augmentation without more
data is a plausible way to slow convergence further, on top of the truncated
LR schedule — consistent with, but not proven by, this single run.

## Bottom line

Bigger model + finer spacing + heavier augmentation, stopped at 100 of a
300-epoch planned schedule, currently underperforms the smaller CPU baseline
by more than half (0.2127 vs. 0.4603). The baseline checkpoint stays in
production. This result mostly demonstrates a truncated, not-yet-converged
run rather than disproving the bigger architecture — a fair test would need
the full 300 epochs (or a schedule matched to whatever epoch budget is
actually available) before drawing conclusions about the architecture itself.
