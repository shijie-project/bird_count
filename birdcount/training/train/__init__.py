"""Training the density model.

config.py        TrainConfig: every hyperparameter, the CLI generated from it
losses.py, ot.py DM-Count loss (batched Sinkhorn) + Gaussian-density aux term
ema.py           EMA of the weights
hard_mining.py   per-image error tracking -> sampling weights
checkpoints.py   atomic writes, best-checkpoint rotation
trainer.py       the loop; validation through `training.eval`, as deployed
sweep.py         grid sweeps and their leaderboard
"""
