"""The ported backbone must be torchvision's ShuffleNetV2-1.0x, layer for layer.

Same parameter names (so checkpoints and ImageNet weights load) and same
outputs given the same weights.
"""

import pytest
import torch

from birdcount.core.model.backbone import IMAGENET_WEIGHTS_URL, ShuffleNetV2Backbone


tv_models = pytest.importorskip("torchvision.models")

_PARTS = ("conv1.", "stage2.", "stage3.")


@pytest.fixture(scope="module")
def reference():
    torch.manual_seed(0)
    return tv_models.shufflenet_v2_x1_0(weights=None).eval()


def test_parameter_names_match_torchvision(reference) -> None:
    ours = ShuffleNetV2Backbone().state_dict()
    theirs = {k: v for k, v in reference.state_dict().items() if k.startswith(_PARTS)}
    assert list(ours) == list(theirs)
    assert all(ours[k].shape == theirs[k].shape for k in ours)


def test_outputs_match_torchvision(reference) -> None:
    ours = ShuffleNetV2Backbone().eval()
    ours.load_state_dict({k: v for k, v in reference.state_dict().items() if k.startswith(_PARTS)})
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        f2, f3 = ours(x)
        ref2 = reference.stage2(reference.maxpool(reference.conv1(x)))
        ref3 = reference.stage3(ref2)
    torch.testing.assert_close(f2, ref2, rtol=0, atol=0)
    torch.testing.assert_close(f3, ref3, rtol=0, atol=0)


def test_weights_url_is_torchvisions() -> None:
    assert tv_models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1.url == IMAGENET_WEIGHTS_URL
