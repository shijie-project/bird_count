"""Compression-robust training: count a low-bitrate stream's frames as the clean frames are counted.

The model reads fine texture, so a frame squeezed through a capped H.264
stream (or scaled down) reads low: at 3 fps / 400 kbps the production model
undercounted crowded floors by ~5 %. Thin links (the office reaching the farm
over a ~10 Mbps tunnel) only carry such streams.

The fix is self-distillation, no new labels: the model as it is (the teacher)
counts clean recording frames; the student sees the same frames as a stream
delivers them and learns to give the teacher's density map. Training mixes
these pairs into the usual labelled batches (`train --distill`), so the clean
counts, and with them the cameras' thresholds, stay where they were.

    pairs.py     windows of the replayed recordings -> (delivered frame, teacher density) pairs
    dataset.py   crops of the pairs for training, and the per-setting report on held-out pairs
"""
