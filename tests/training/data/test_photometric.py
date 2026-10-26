import pytest
import torch

from birdcount.core.inference import IMAGENET_MEAN, IMAGENET_STD
from birdcount.training.data import photometric as P


tvf = pytest.importorskip("torchvision.transforms.functional")


@pytest.fixture
def batch() -> torch.Tensor:
    return torch.rand(4, 3, 32, 48, generator=torch.Generator().manual_seed(0))


def _factors(n: int, lo: float, hi: float) -> torch.Tensor:
    return torch.linspace(lo, hi, n).view(n, 1, 1, 1)


@pytest.mark.parametrize(
    ("ours", "theirs", "lo", "hi"),
    [
        (P.adjust_brightness, tvf.adjust_brightness, 0.6, 1.4),
        (P.adjust_contrast, tvf.adjust_contrast, 0.6, 1.4),
        (P.adjust_saturation, tvf.adjust_saturation, 0.7, 1.3),
        (P.adjust_hue, tvf.adjust_hue, -0.1, 0.1),
    ],
)
def test_matches_torchvision_per_sample(batch, ours, theirs, lo, hi) -> None:
    f = _factors(len(batch), lo, hi)
    got = ours(batch.clone(), f)
    want = torch.stack([theirs(img, float(fi)) for img, fi in zip(batch, f.flatten(), strict=True)])
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_without_augmentation_it_is_bgr_to_normalised_rgb() -> None:
    bgr = torch.randint(0, 256, (2, 16, 24, 3), dtype=torch.uint8)
    out = P.Photometric()(bgr, augment=False)
    rgb = bgr.permute(0, 3, 1, 2).flip(1).float() / 255
    want = (rgb - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    torch.testing.assert_close(out, want)


def test_augmentation_is_reproducible_and_finite() -> None:
    bgr = torch.randint(0, 256, (4, 32, 32, 3), dtype=torch.uint8)
    aug = P.Photometric()
    a = aug(bgr, augment=True, generator=torch.Generator().manual_seed(9))
    b = aug(bgr, augment=True, generator=torch.Generator().manual_seed(9))
    c = aug(bgr, augment=True, generator=torch.Generator().manual_seed(10))
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)
    assert torch.isfinite(a).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_runs_on_cuda() -> None:
    bgr = torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8, device="cuda")
    out = P.Photometric()(bgr, augment=True, generator=torch.Generator().manual_seed(0))
    assert out.device.type == "cuda" and out.shape == (4, 3, 64, 64)
