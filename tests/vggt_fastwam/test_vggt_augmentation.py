from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from fastwam.aug import DomainRandomization
from fastwam.datasets.lerobot.processors.fastwam_processor import prepare_vggt_images


ROOT = Path(__file__).resolve().parents[2]
TASK = "e05"
KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def _config(task=TASK):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="train", overrides=[f"task={task}"])


def _processor(mode="missing"):
    cfg = _config()
    node = cfg.data.train.processor
    if mode == "missing":
        del node.vggt_augmentation
    else:
        node.vggt_augmentation.enabled = mode != "disabled"
        node.vggt_augmentation.p = 0.0 if mode == "zero" else 1.0
    processor = instantiate(node).train()
    # Exercise actual image/action assembly without requiring dataset statistics.
    processor._normalizer = SimpleNamespace(forward=lambda data: data)
    return processor


@pytest.fixture
def raw_sample():
    generator = torch.Generator().manual_seed(17)
    return {
        "images": {key: torch.randint(256, (1, 3, 240, 320), dtype=torch.uint8,
                                      generator=generator).expand(33, -1, -1, -1).clone()
                   for key in KEYS},
        "action": {"default": torch.zeros(32, 14)},
        "state": {"default": torch.zeros(33, 14)},
        "image_is_pad": torch.zeros(33, dtype=torch.bool),
        "action_is_pad": torch.zeros(32, dtype=torch.bool),
        "state_is_pad": torch.zeros(33, dtype=torch.bool),
        "action_dim_is_pad": torch.zeros(14, dtype=torch.bool),
        "state_dim_is_pad": torch.zeros(14, dtype=torch.bool),
        "idx": 0,
        "task": "pick up the object",
        "vggt_recon_depth": torch.ones(3, 240, 320),
        "vggt_recon_depth_valid": torch.ones(3, 240, 320, dtype=torch.bool),
        "vggt_recon_intrinsics": torch.eye(3).repeat(3, 1, 1),
        "vggt_recon_extrinsics": torch.eye(4)[:3].repeat(3, 1, 1),
    }


def _run(processor, sample):
    torch.manual_seed(42)
    np.random.seed(42)
    output = processor.preprocess(deepcopy(sample))
    return output, torch.get_rng_state(), np.random.get_state()


@pytest.mark.parametrize("mode", ["disabled", "zero"])
def test_disabled_processor_preserves_old_outputs_and_rng(raw_sample, mode):
    old, old_rng, old_numpy = _run(_processor(), raw_sample)
    actual, rng, numpy_rng = _run(_processor(mode), raw_sample)
    assert actual.keys() == old.keys()
    for key in old:
        if isinstance(old[key], torch.Tensor):
            assert torch.equal(old[key], actual[key]), key
        else:
            assert old[key] == actual[key], key
    assert torch.equal(rng, old_rng)
    np.testing.assert_equal(numpy_rng, old_numpy)
    assert torch.equal(actual["vggt_images"], prepare_vggt_images(raw_sample["images"], KEYS, 256))


def test_enabled_processor_changes_only_current_rgb(raw_sample):
    baseline, _, _ = _run(_processor(), raw_sample)
    processor = _processor("enabled")
    actual, _, _ = _run(processor, raw_sample)
    images = actual["vggt_images"]
    assert images.shape == (3, 3, 224, 288)
    assert images.dtype == torch.float32
    assert torch.isfinite(images).all() and images.min() >= 0 and images.max() <= 1
    assert all(not torch.equal(images[v], baseline["vggt_images"][v]) for v in range(3))
    for key, expected in baseline.items():
        if key != "vggt_images":
            if isinstance(expected, torch.Tensor):
                assert torch.equal(actual[key], expected), key
            else:
                assert actual[key] == expected, key


def test_eval_bypasses_enabled_augmentation(raw_sample):
    class FailIfCalled:
        def __call__(self, _):
            raise AssertionError("Evaluation called augmentation")

    processor = _processor("enabled").eval()
    processor.vggt_augmentation = FailIfCalled()
    actual, _, _ = _run(processor, raw_sample)
    assert torch.equal(actual["vggt_images"], prepare_vggt_images(raw_sample["images"], KEYS, 256))


@pytest.mark.parametrize("kwargs", [{}, {"enabled": False}, {"enabled": True, "p": 0}])
def test_noop_construction_and_call_do_not_consume_rng(kwargs):
    views = torch.full((3, 1, 3, 24, 32), 0.5)
    before = torch.get_rng_state()
    augmentation = DomainRandomization(**kwargs)
    assert augmentation(views) is views
    assert torch.equal(before, torch.get_rng_state())


def test_independent_cameras_temporally_shared_and_no_input_mutation():
    torch.manual_seed(9)
    frame = torch.rand(1, 1, 3, 24, 32)
    views = frame.expand(3, 2, -1, -1, -1).clone()
    before = views.clone()
    augmentation = DomainRandomization(enabled=True, p=1,
        style={"enabled": True, "p": 1}, fourier={"enabled": True, "p": 1})
    result = augmentation(views)
    assert torch.equal(views, before)
    assert all(torch.equal(result[v, 0], result[v, 1]) for v in range(3))
    assert not torch.equal(result[0], result[1])
    assert not torch.equal(result[1], result[2])
    torch.manual_seed(123)
    first = augmentation(views)
    torch.manual_seed(123)
    assert torch.equal(first, augmentation(views))


def test_sample_gate_and_stage_order(monkeypatch):
    augmentation = DomainRandomization(enabled=True, p=0.8)
    events = []
    for method, label in (("_lighting", "photo"), ("_channel_statistics", "style"), ("_spectrum", "fourier")):
        def stage(frames, label=label):
            events.append(label)
            return frames
        monkeypatch.setattr(augmentation, method, stage)
    calls = []
    def gate(*args, **kwargs):
        calls.append(1)
        return torch.tensor(0.1)
    monkeypatch.setattr(torch, "rand", gate)
    views = torch.full((3, 1, 3, 24, 32), 0.5)
    augmentation(views)
    assert calls == [1]
    assert events == ["photo", "style", "fourier"] * 3
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.9))
    events.clear()
    assert augmentation(views) is views
    assert events == []


@pytest.mark.parametrize("task", [TASK])
def test_release_task_preserves_enabled_augmentation(task):
    cfg = _config(task)
    assert cfg.data.train.processor.vggt_augmentation.enabled is True
    cfg.data.train.processor.vggt_augmentation.enabled = True
    processor = instantiate(cfg.data.train.processor)
    assert processor.vggt_augmentation.enabled is True
    assert processor.vggt_augmentation.p == 0.8
    assert processor.vggt_augmentation.style["enabled"] is True
    assert processor.vggt_augmentation.fourier["enabled"] is True


def test_deployment_does_not_augment_with_enabled_training_config():
    from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy, _ROBOTWIN_CAMERA_MAP

    processor = _processor("enabled").eval()
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.processor = processor
    rgb = np.arange(240 * 320 * 3, dtype=np.int64).reshape(240, 320, 3).astype(np.uint8)
    observation = {"observation": {_ROBOTWIN_CAMERA_MAP[key]: {"rgb": rgb.copy()} for key in KEYS}}
    raw = {key: torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0) for key in KEYS}
    rng = torch.get_rng_state()
    actual = policy._build_robotwin_vggt_tensor(observation)
    assert torch.equal(actual, prepare_vggt_images(raw, KEYS, 256).unsqueeze(0))
    assert torch.equal(rng, torch.get_rng_state())
