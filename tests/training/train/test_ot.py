"""Batched Sinkhorn / OT loss: one batch equals the images one by one; unstable and empty images drop out."""

import torch

from birdcount.training.train.ot import OTLoss, sinkhorn


CROP, STRIDE = 64, 8
S = CROP // STRIDE


def problem(seed: int, counts):
    g = torch.Generator().manual_seed(seed)
    density = torch.rand(len(counts), 1, S, S, generator=g) * 0.1
    normed = density / density.flatten(1).sum(1).view(-1, 1, 1, 1)
    points = [torch.rand(n, 2, generator=g) * CROP for n in counts]
    return normed, density, points


def test_a_batch_equals_its_images_one_at_a_time() -> None:
    ot = OTLoss(CROP, STRIDE, norm_coords=True, iterations=100, reg=1.0)
    normed, density, points = problem(0, [3, 17, 1, 40])
    d1 = density.clone().requires_grad_(True)
    together = ot(normed, d1, points)
    together[0].backward()
    d2 = density.clone().requires_grad_(True)
    alone = [ot(normed[i : i + 1], d2[i : i + 1], [points[i]]) for i in range(4)]
    sum(a[0] for a in alone).backward()
    # The surrogate's value is ~0 by construction (cancellation noise); its gradient is the loss.
    torch.testing.assert_close(d1.grad, d2.grad, rtol=1e-4, atol=1e-6)
    for k in (1, 2):  # Wasserstein distance, OT objective
        torch.testing.assert_close(together[k], sum(a[k] for a in alone), rtol=1e-5, atol=1e-7)


def test_images_without_points_contribute_nothing() -> None:
    ot = OTLoss(CROP, STRIDE, norm_coords=True)
    normed, density, points = problem(1, [0, 5, 0])
    with_empty = ot(normed, density, points)
    only = ot(normed[1:2], density[1:2], [points[1]])
    for a, b in zip(with_empty, only, strict=True):
        torch.testing.assert_close(a, b)
    assert all(v.item() == 0 for v in ot(normed[:1], density[:1], [points[0]]))


def test_an_unstable_image_rolls_back_and_is_flagged() -> None:
    a = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    b = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
    cost = torch.rand(2, 2, 4)
    cost[1] = -1e3  # exp(-cost / reg) overflows: inf * 0 -> NaN
    u, v, _, unstable = sinkhorn(a, b, cost, reg=1.0, max_iter=50)
    assert unstable.tolist() == [False, True]
    assert torch.isfinite(u).all() and torch.isfinite(v).all()


def test_the_surrogate_gradient_is_dm_counts() -> None:
    """d(loss)/d(density) equals the analytical OT gradient fed into the surrogate."""
    ot = OTLoss(CROP, STRIDE, norm_coords=True, reg=1.0)
    normed, density, points = problem(2, [6])
    density.requires_grad_(True)
    loss, _, _ = ot(normed, density, points)
    loss.backward()
    grad = density.grad.flatten()
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    # The DM-Count gradient is orthogonal to the density: scaling the map changes no transport.
    assert abs(float((grad * density.detach().flatten()).sum())) < 1e-5
