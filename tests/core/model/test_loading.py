import argparse
import pickle

import pytest
import torch
from torch import nn

from birdcount.core.model import DensityNet, build_model, load_inference_model, read_state_dict


@pytest.fixture(scope="module")
def weights() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return DensityNet().state_dict()


def _save(tmp_path, obj, name="ckpt.pth"):
    path = tmp_path / name
    torch.save(obj, path)
    return path


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_raw_state_dict(tmp_path, weights) -> None:
    assert _same(read_state_dict(_save(tmp_path, weights)), weights)


def test_training_checkpoint_prefers_ema(tmp_path, weights) -> None:
    ema = {k: v + 1 if v.is_floating_point() else v for k, v in weights.items()}
    ckpt = {"epoch": 3, "model_state_dict": weights, "optimizer_state_dict": {}, "ema_state_dict": ema}
    assert _same(read_state_dict(_save(tmp_path, ckpt, "ckpt.tar")), ema)


def test_training_checkpoint_without_ema(tmp_path, weights) -> None:
    ckpt = {"epoch": 3, "model_state_dict": weights, "ema_state_dict": None}
    assert _same(read_state_dict(_save(tmp_path, ckpt)), weights)


def test_data_parallel_prefix_is_stripped(tmp_path, weights) -> None:
    prefixed = {f"module.{k}": v for k, v in weights.items()}
    assert _same(read_state_dict(_save(tmp_path, prefixed)), weights)


def test_missing_checkpoint_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_state_dict(tmp_path / "nope.pth")


def test_mismatched_keys_raise(tmp_path, weights) -> None:
    broken = dict(weights)
    broken.pop("density_conv.bias")
    with pytest.raises(RuntimeError, match=r"density_conv\.bias"):
        build_model(_save(tmp_path, broken))


def test_checkpoints_cannot_run_code(tmp_path) -> None:
    # Unpickling an arbitrary class instance can run code; weights_only refuses it.
    ckpt = {"model_state_dict": {}, "args": argparse.Namespace(lr=1e-5)}
    with pytest.raises(pickle.UnpicklingError, match=r"Weights only load failed"):
        read_state_dict(_save(tmp_path, ckpt))


def test_checkpoint_and_pretrained_are_exclusive(tmp_path, weights) -> None:
    with pytest.raises(ValueError):
        build_model(_save(tmp_path, weights), pretrained_backbone=True)


def test_build_model_loads_weights_in_train_mode(tmp_path, weights) -> None:
    m = build_model(_save(tmp_path, weights))
    assert m.training
    assert _same(m.state_dict(), weights)


def test_inference_model_on_cpu(tmp_path, weights) -> None:
    m = load_inference_model(_save(tmp_path, weights), "cpu", input_order="bgr")
    assert not m.training
    assert m.input_order == "bgr"
    assert not any(isinstance(x, nn.BatchNorm2d) for x in m.modules())
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in m.parameters())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_inference_model_on_cuda_is_half(tmp_path, weights) -> None:
    m = load_inference_model(_save(tmp_path, weights), "cuda")
    assert all(p.dtype == torch.float16 and p.is_cuda for p in m.parameters())
    with torch.inference_mode():
        out = m(torch.randn(1, 3, 64, 96, device="cuda", dtype=torch.float16))
    assert out.shape == (1, 1, 8, 12)
