"""Training and evaluation data: annotations, the decoded-frame cache,
augmentation (geometric on CPU workers, photometric on the GPU), sampling.

Every image goes through `core.frames.fit_frame` before anything else, so
training, evaluation and deployment all see pixels prepared the same way.
"""
