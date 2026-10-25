import pytest
import torch
from torch import nn

from birdcount.core.model import OUTPUT_STRIDE, DensityNet


def _randomize_batchnorm(model: nn.Module) -> None:
    """Fresh BN layers are identities (mean 0, var 1), which would make a
    fusion test pass vacuously."""
    g = torch.Generator().manual_seed(0)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            n = m.num_features
            m.running_mean.copy_(torch.randn(n, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(n, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(n, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(n, generator=g) * 0.1)


@pytest.fixture
def model() -> DensityNet:
    torch.manual_seed(0)
    m = DensityNet()
    _randomize_batchnorm(m)
    # Bias the output conv so the ReLU'd density is not all zeros.
    m.density_conv.bias.data.fill_(0.5)
    return m.eval()


@pytest.mark.parametrize(("h", "w"), [(64, 96), (72, 136)])  # 72x136 -> odd-sized stage2 (9x17)
def test_output_is_input_over_stride(model, h, w) -> None:
    with torch.no_grad():
        out = model(torch.randn(2, 3, h, w))
    assert out.shape == (2, 1, h // OUTPUT_STRIDE, w // OUTPUT_STRIDE)
    assert (out >= 0).all()


def test_fusion_is_exact_and_complete(model) -> None:
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        before = model(x)
        model.fuse_conv_bn()
        after = model(x)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert before.abs().max() > 0
    torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-5)


def test_fusion_refuses_train_mode(model) -> None:
    with pytest.raises(RuntimeError, match="eval mode"):
        model.train().fuse_conv_bn()


def test_bgr_input_is_equivalent_and_idempotent(model) -> None:
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        rgb = model(x)
        model.set_input_order("bgr")
        model.set_input_order("bgr")  # second call must be a no-op
        bgr = model(x[:, [2, 1, 0]])
        model.set_input_order("rgb")
        back = model(x)
    torch.testing.assert_close(bgr, rgb)
    torch.testing.assert_close(back, rgb)


def test_bgr_survives_fusion(model) -> None:
    x = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        rgb = model(x)
        model.fuse_conv_bn()
        model.set_input_order("bgr")
        bgr = model(x[:, [2, 1, 0]])
    torch.testing.assert_close(bgr, rgb, rtol=1e-4, atol=1e-5)


def test_invalid_input_order() -> None:
    with pytest.raises(ValueError):
        DensityNet().set_input_order("hsv")


@pytest.mark.parametrize("frozen", [True, False])
def test_backbone_bn_freezing(frozen) -> None:
    m = DensityNet(freeze_backbone_bn=frozen).train()
    backbone_bn = [x for x in m.backbone.modules() if isinstance(x, nn.BatchNorm2d)]
    head_bn = [x for x in m.reg_layer.modules() if isinstance(x, nn.BatchNorm2d)]
    assert all(bn.training != frozen for bn in backbone_bn)
    assert all(bn.training for bn in head_bn)
