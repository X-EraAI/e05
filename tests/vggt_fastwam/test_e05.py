import pytest
import torch
import torch.nn as nn

from fastwam.models.wan22.vggt_lang_recon_backbone import (
    VGGTLanguageReconBackbone,
)


class CountingPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 64, kernel_size=2, stride=2)
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return self.proj(images).flatten(2).transpose(1, 2)


class TinyCameraHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 9)

    def forward(self, features, patch_token_start):
        pose = self.proj(features[-1][:, :, 0])
        return torch.cat((pose[..., :7], pose[..., 7:].sigmoid() + 0.5), dim=-1)


class TinyDenseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 1)

    def forward(self, features, images, patch_token_start):
        batch, views, _, _ = features[-1].shape
        height, width = images.shape[-2:]
        patch_h, patch_w = height // 2, width // 2
        patches = self.proj(features[-1][:, :, patch_token_start:]).squeeze(-1)
        patches = patches.reshape(batch * views, patch_h, patch_w)
        depth = patches.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        depth = depth.reshape(batch, views, height, width).float().abs() + 1.0
        return depth, depth + 1.0


def _tiny_backbone(*, trainable=True):
    from vggt_omega.models.aggregator import Aggregator

    aggregator = Aggregator(
        patch_size=2,
        embed_dim=64,
        depth=2,
        num_heads=2,
        mlp_ratio=2,
        num_register_tokens=1,
        register_attention_block_indices=[1],
        cached_layer_indices=(0, 1),
        patch_embed=CountingPatchEmbed(),
        callback_layer_indices=(0, 1),
    )
    for module in aggregator.modules():
        if hasattr(module, "bias_mask"):
            module.bias_mask.fill_(1)
        if hasattr(module, "gamma") and hasattr(module, "init_values"):
            nn.init.constant_(module.gamma, module.init_values)
    return VGGTLanguageReconBackbone(
        aggregator,
        callback_layer_indices=(0, 1),
        text_dim=10,
        trainable=trainable,
        camera_head=TinyCameraHead(),
        dense_head=TinyDenseHead(),
    )


def test_language_recon_backbone_shapes_and_single_tokenizer_call():
    model = _tiny_backbone().eval()
    images = torch.rand(2, 2, 3, 4, 4)
    context = torch.rand(2, 5, 10)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)

    output = model.forward_with_reconstruction(
        images, context=context, language_context_mask=mask
    )

    assert model.aggregator.patch_embed.calls == 1
    assert output["future_depth"].shape == (2, 2, 4, 4)
    assert output["future_depth_conf"].shape == (2, 2, 4, 4)
    assert output["future_pose"].shape == (2, 2, 9)
    assert set(output["kv_by_layer"]) == {0, 1}
    assert output["kv_by_layer"][0]["k"].shape == (2, 2, 12, 32)
    assert output["kv_by_layer"][1]["k"].shape == (2, 2, 4, 32)


def test_padding_language_tokens_are_ignored_but_valid_language_changes_visual_kv():
    torch.manual_seed(7)
    model = _tiny_backbone().eval()
    images = torch.rand(1, 2, 3, 4, 4)
    context = torch.rand(1, 5, 10)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

    baseline = model(images, context=context, language_context_mask=mask)
    padded_changed = context.clone()
    padded_changed[:, 3:] = float("nan")
    ignored = model(images, context=padded_changed, language_context_mask=mask)
    for layer in baseline:
        torch.testing.assert_close(baseline[layer]["k"], ignored[layer]["k"])
        torch.testing.assert_close(baseline[layer]["v"], ignored[layer]["v"])

    valid_changed = context.clone()
    valid_changed[:, 0] += 1.0
    changed = model(images, context=valid_changed, language_context_mask=mask)
    assert any(
        not torch.allclose(baseline[layer]["k"], changed[layer]["k"])
        for layer in baseline
    )


def test_language_projector_and_visual_vggt_receive_gradients():
    model = _tiny_backbone(trainable=True).train()
    images = torch.rand(1, 2, 3, 4, 4)
    context = torch.rand(1, 3, 10)
    mask = torch.ones(1, 3, dtype=torch.bool)
    output = model.forward_with_reconstruction(
        images, context=context, language_context_mask=mask
    )
    loss = output["future_depth"].mean()
    loss = loss + output["future_pose"].mean()
    loss = loss + sum(layer["v"].mean() for layer in output["kv_by_layer"].values())
    loss.backward()

    projector_grad = model.language_projector[0].weight.grad
    visual_grad = model.aggregator.inter_frame_blocks[0].attn.qkv.weight.grad
    assert projector_grad is not None and torch.isfinite(projector_grad).all()
    assert visual_grad is not None and torch.isfinite(visual_grad).all()
    for head in (model.camera_head, model.dense_head):
        assert head.proj.weight.grad is not None
        assert torch.isfinite(head.proj.weight.grad).all()


def test_invalid_language_context_shapes_are_rejected():
    model = _tiny_backbone().eval()
    images = torch.rand(1, 2, 3, 4, 4)
    with pytest.raises(ValueError, match="language_context_mask"):
        model(
            images,
            context=torch.rand(1, 3, 10),
            language_context_mask=torch.ones(1, 2, dtype=torch.bool),
        )


def _tiny_model(*, action_dim=2, proprio_dim=None, train_vggt=True, projection_sharing="shared", reconstruction_target="future"):
    from fastwam.models.wan22.action_dit import ActionDiT
    from e05 import E05
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.vggt_bridge import VGGTSharedAttentionBridge

    action = ActionDiT(hidden_dim=8, action_dim=action_dim, ffn_dim=16, text_dim=10,
                       freq_dim=8, eps=1e-6, num_heads=2, attn_head_dim=4,
                       num_layers=2, use_gradient_checkpointing=False)
    bridge = VGGTSharedAttentionBridge(
        fastwam_width=8, vggt_width=64, fastwam_num_heads=2, vggt_num_heads=2,
        layer_pairs=((0, 0), (1, 1)), target_experts=("action",),
        projection_mode="vggt_to_fastwam",
        projection_sharing=projection_sharing,
    )
    mot = MoT(mixtures={"action": action}, mot_checkpoint_mixed_attn=False,
              vggt_bridge=bridge)
    return E05(
        action_expert=action, mot=mot,
        vggt_backbone=_tiny_backbone(trainable=train_vggt), text_dim=10,
        train_vggt=train_vggt, action_num_train_timesteps=10,
        proprio_dim=proprio_dim, reconstruction_target=reconstruction_target,
        reconstruction_loss_config={"gradient_weight": 0.5, "use_conf_loss": False},
    )


def _future_sample(batch=2):
    return {
        "action": torch.rand(batch, 4, 2),
        "context": torch.rand(batch, 3, 10),
        "context_mask": torch.ones(batch, 3, dtype=torch.bool),
        "language_context_mask": torch.ones(batch, 3, dtype=torch.bool),
        "vggt_images": torch.rand(batch, 2, 3, 4, 4),
        "vggt_recon_future_depth": torch.ones(batch, 2, 4, 4),
        "vggt_recon_future_depth_valid": torch.ones(batch, 2, 4, 4, dtype=torch.bool),
        "vggt_recon_future_intrinsics": torch.eye(3).repeat(batch, 2, 1, 1),
        "vggt_recon_future_extrinsics": torch.eye(4)[:3].repeat(batch, 2, 1, 1),
    }


def _prediction(batch=2):
    pose = torch.zeros(batch, 2, 9)
    pose[..., 3] = 1
    pose[..., 7:] = 1
    return {
        "future_depth": torch.ones(batch, 2, 4, 4, requires_grad=True),
        "future_depth_conf": torch.ones(batch, 2, 4, 4, requires_grad=True),
        "future_pose": pose.requires_grad_(),
    }


def test_future_padding_excludes_nan_labels_and_preserves_all_head_gradients():
    model = _tiny_model()
    sample = _future_sample()
    prediction = _prediction()
    keep = torch.tensor([True, False])
    expected, expected_metrics = model._future_recon_loss(prediction, sample, keep)
    for key in ("vggt_recon_future_depth", "vggt_recon_future_intrinsics",
                "vggt_recon_future_extrinsics"):
        sample[key][1] = float("nan")
    for value in prediction.values():
        with torch.no_grad():
            value[1] = float("nan")
    loss, metrics = model._future_recon_loss(prediction, sample, keep)
    torch.testing.assert_close(loss, expected)
    assert set(metrics) == set(expected_metrics)
    loss.backward()
    for value in prediction.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[1]) == 0

    # No labels may be read when the entire batch is padded. The empty
    # reductions also keep camera/depth/conf branches connected on this rank.
    prediction = {k: v.detach().requires_grad_() for k, v in prediction.items()}
    zero, zero_metrics = model._future_recon_loss(prediction, {}, ~torch.ones(2, dtype=torch.bool))
    assert zero.item() == 0 and set(zero_metrics) == set(metrics)
    zero.backward()
    for value in prediction.values():
        assert value.grad is not None and torch.count_nonzero(value.grad) == 0


def test_training_inference_and_checkpoint_round_trip(tmp_path):
    model = _tiny_model()
    sample = _future_sample(batch=1)
    loss, metrics = model.training_loss(sample)
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in metrics.values())
    loss.backward()
    assert model.vggt_backbone.language_projector[0].weight.grad is not None
    checkpoint = tmp_path / "model.pt"
    model.save_checkpoint(checkpoint, step=2)
    restored = _tiny_model()
    assert restored.load_checkpoint(checkpoint)["step"] == 2
    kwargs = dict(action_horizon=4, vggt_images=sample["vggt_images"],
                  context=sample["context"], context_mask=sample["context_mask"],
                  language_context_mask=sample["language_context_mask"],
                  return_reconstruction=True, num_inference_steps=1, seed=7)
    first = model.infer_action(**kwargs)
    second = restored.infer_action(**kwargs)
    torch.testing.assert_close(first["action"], second["action"])
    for key in ("depth", "confidence", "pose"):
        torch.testing.assert_close(first["reconstruction"]["future"][key],
                                   second["reconstruction"]["future"][key])
    payload = torch.load(checkpoint, weights_only=True)
    payload["metadata"]["future_reconstruction"] = False
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="metadata mismatch"):
        restored.load_checkpoint(checkpoint)
    payload["metadata"]["future_reconstruction"] = True
    del payload["vggt_backbone"]["language_projector.0.weight"]
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="language_projector"):
        restored.load_checkpoint(checkpoint)


def test_pretrained_16_to_14_transfer_training_and_eval(tmp_path):
    source = _tiny_model(action_dim=16, proprio_dim=16)
    checkpoint = tmp_path / "pretrained.pt"
    source.save_checkpoint(checkpoint, step=123)
    target = _tiny_model(action_dim=14, proprio_dim=14)
    target.vggt_camera_keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    target.lambda_vggt_recon = 0.01
    report = target.load_pretrained_checkpoint(
        checkpoint, reset_action_projections=True, reset_proprio_encoder=True
    )
    assert (report["source_action_dim"], report["target_action_dim"]) == (16, 14)
    assert (report["source_proprio_dim"], report["target_proprio_dim"]) == (16, 14)
    assert len(report["reset_keys"]) == 6
    for name, source_module, target_module in (
        ("action_expert", source.action_expert, target.action_expert),
        ("vggt_bridge", source.mot.vggt_bridge, target.mot.vggt_bridge),
        ("vggt_backbone", source.vggt_backbone, target.vggt_backbone),
    ):
        for key, value in source_module.state_dict().items():
            if f"{name}.{key}" not in report["reset_keys"]:
                torch.testing.assert_close(value, target_module.state_dict()[key], rtol=0, atol=0)

    assert target.action_expert.action_encoder.weight.shape == (8, 14)
    assert target.action_expert.head.weight.shape == (14, 8)
    assert target.proprio_encoder.weight.shape == (10, 14)
    sample = _future_sample(batch=1)
    sample["action"] = torch.rand(1, 4, 14)
    sample["proprio"] = torch.rand(1, 14)
    optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    before = target.action_expert.action_encoder.weight.detach().clone()
    loss, metrics = target.training_loss(sample)
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in metrics.values())
    loss.backward()
    for module in (target.action_expert.action_encoder, target.action_expert.head,
                   target.proprio_encoder):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, target.action_expert.action_encoder.weight)

    finetuned = tmp_path / "finetuned.pt"
    target.save_checkpoint(finetuned, step=1)
    restored = _tiny_model(action_dim=14, proprio_dim=14)
    restored.vggt_camera_keys = target.vggt_camera_keys
    restored.lambda_vggt_recon = target.lambda_vggt_recon
    checkpoint.unlink()  # Evaluation must be independent of the pretrain file.
    restored.load_checkpoint(finetuned)
    kwargs = dict(action_horizon=4, vggt_images=sample["vggt_images"],
                  context=sample["context"], context_mask=sample["context_mask"],
                  language_context_mask=sample["language_context_mask"],
                  proprio=sample["proprio"], num_inference_steps=1, seed=7,
                  return_reconstruction=True)
    first = target.infer_action(**kwargs)
    second = restored.infer_action(**kwargs)
    assert first["action"].shape == (1, 4, 14)
    assert torch.isfinite(first["action"]).all()
    torch.testing.assert_close(first["action"], second["action"])


def test_pretrained_same_dimensions_reuses_all_weights(tmp_path):
    source = _tiny_model(action_dim=14, proprio_dim=14)
    checkpoint = tmp_path / "source.pt"
    source.save_checkpoint(checkpoint)
    target = _tiny_model(action_dim=14, proprio_dim=14)
    report = target.load_pretrained_checkpoint(checkpoint)
    assert report["reset_keys"] == []
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("load_action,load_bridge", [(False, False), (False, True), (True, False)])
def test_pretrained_selective_transfer_preserves_fresh_components(tmp_path, load_action, load_bridge):
    source = _tiny_model(action_dim=16, proprio_dim=16, projection_sharing="independent")
    # Make every checkpoint tensor distinguishable, including normally constant
    # norm weights and block modulation parameters.
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(0.125)
    checkpoint = tmp_path / "source.pt"
    source.save_checkpoint(checkpoint)
    target = _tiny_model(action_dim=14, proprio_dim=14, projection_sharing="independent")
    components = {
        "action_expert": target.action_expert,
        "vggt_bridge": target.mot.vggt_bridge,
        "vggt_backbone": target.vggt_backbone,
    }
    initial = {name: {key: value.clone() for key, value in module.state_dict().items()}
               for name, module in components.items()}
    parameter_ids = [id(parameter) for parameter in target.parameters()]
    report = target.load_pretrained_checkpoint(
        checkpoint, reset_action_projections=True, reset_proprio_encoder=True,
        load_pretrained_action_expert=load_action, load_pretrained_bridge=load_bridge,
    )
    assert [id(parameter) for parameter in target.parameters()] == parameter_ids
    payload = torch.load(checkpoint, weights_only=True)
    retained = {name for name, load in (("action_expert", load_action),
                                      ("vggt_bridge", load_bridge)) if not load}
    assert set(report["retained_keys"]) == {
        f"{name}.{key}" for name in retained for key in initial[name]
    }
    assert not set(report["retained_keys"]) & set(report["loaded_keys"])
    for name, module in components.items():
        for key, value in module.state_dict().items():
            if name in retained:
                torch.testing.assert_close(value, initial[name][key], rtol=0, atol=0)
            elif f"{name}.{key}" not in report["reset_keys"]:
                torch.testing.assert_close(value, payload[name][key], rtol=0, atol=0)

    sample = _future_sample(batch=1)
    sample["action"] = torch.rand(1, 4, 14)
    sample["proprio"] = torch.rand(1, 14)
    loss, _ = target.training_loss(sample)
    assert torch.isfinite(loss)
    loss.backward()
    for module in (target.action_expert.action_encoder, target.action_expert.head,
                   target.action_expert.blocks[0].self_attn.q,
                   target.mot.vggt_bridge, target.vggt_backbone.language_projector):
        grads = [parameter.grad for parameter in module.parameters()]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
        assert any(grad.abs().sum() > 0 for grad in grads)

    # Strict restoration must restore the saved action/bridge weights normally.
    finetuned = tmp_path / "finetuned.pt"
    target.save_checkpoint(finetuned)
    restored = _tiny_model(action_dim=14, proprio_dim=14, projection_sharing="independent")
    restored.load_checkpoint(finetuned)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_selective_transfer_validates_backbone_before_mutating(tmp_path):
    checkpoint = tmp_path / "source.pt"
    _tiny_model(action_dim=16, proprio_dim=16).save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    del payload["vggt_backbone"]["language_projector.0.weight"]
    torch.save(payload, checkpoint)
    target = _tiny_model(action_dim=14, proprio_dim=14)
    initial = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="vggt_backbone key mismatch"):
        target.load_pretrained_checkpoint(
            checkpoint, load_pretrained_action_expert=False, load_pretrained_bridge=False,
            reset_action_projections=True, reset_proprio_encoder=True,
        )
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["action_dim", "proprio_dim", "missing", "shape", "metadata"])
def test_pretrained_rejects_incompatible_weights_before_mutating(tmp_path, corruption):
    source = _tiny_model(action_dim=16, proprio_dim=16)
    checkpoint = tmp_path / "source.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    if corruption == "missing":
        del payload["vggt_backbone"]["language_projector.0.weight"]
    elif corruption == "shape":
        payload["vggt_backbone"]["language_projector.0.weight"] = torch.zeros(1, 1)
    elif corruption == "metadata":
        payload["metadata"]["bridge_config"]["projection_sharing"] = "incompatible"
    torch.save(payload, checkpoint)
    target = _tiny_model(action_dim=14, proprio_dim=14)
    original = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="mismatch"):
        target.load_pretrained_checkpoint(
            checkpoint, reset_action_projections=corruption != "action_dim",
            reset_proprio_encoder=corruption != "proprio_dim",
        )
    for key, value in original.items():
        torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)
    if corruption in {"action_dim", "proprio_dim"}:
        with pytest.raises(RuntimeError, match="size mismatch"):
            target.load_checkpoint(checkpoint)






@pytest.mark.parametrize("load_components", [True, False])
def test_runtime_full_pretrain_skips_component_files(tmp_path, monkeypatch, load_components):
    from e05 import create_e05
    from fastwam.models.wan22 import vggt_lang_recon_backbone

    checkpoint = tmp_path / "source.pt"
    _tiny_model(action_dim=16, proprio_dim=16).save_checkpoint(checkpoint)

    def build_backbone(**kwargs):
        assert kwargs["skip_vggt_load_from_pretrain"] is True
        return _tiny_backbone()

    monkeypatch.setattr(vggt_lang_recon_backbone, "build_vggt_language_recon_backbone", build_backbone)
    model = create_e05(
        vggt_weights_path="/missing/vggt.pt",
        action_dit_pretrained_path="/missing/action.pt",
        pretrained_checkpoint_path=str(checkpoint),
        reset_action_projections=True, reset_proprio_encoder=True,
        load_pretrained_action_expert=load_components,
        load_pretrained_bridge=load_components,
        action_dit_config=dict(hidden_dim=8, action_dim=14, ffn_dim=16, text_dim=10,
                               freq_dim=8, eps=1e-6, num_heads=2, attn_head_dim=4,
                               num_layers=2, use_gradient_checkpointing=False),
        action_scheduler=dict(train_shift=5, infer_shift=5, num_train_timesteps=10),
        layer_pairs=((0, 0), (1, 1)), projection_mode="vggt_to_fastwam",
        vggt_width=64, vggt_num_heads=2, proprio_dim=14,
        reconstruction_loss_config={"gradient_weight": 0.5, "use_conf_loss": False},
        device="cpu", model_dtype=torch.float32,
    )
    assert model.pretrained_load_report["source_action_dim"] == 16
    assert model.pretrained_load_report["target_action_dim"] == 14
    assert model.pretrained_load_report["target_proprio_dim"] == 14
    report = model.pretrained_load_report
    for name in ("action_expert", "vggt_bridge"):
        assert any(key.startswith(f"{name}.") for key in report["loaded_keys"]) == load_components
        assert any(key.startswith(f"{name}.") for key in report["retained_keys"]) != load_components


@pytest.mark.parametrize("source_dim,target_dim", [(None, 14), (16, None)])
def test_pretrained_can_add_or_remove_proprio(tmp_path, source_dim, target_dim):
    checkpoint = tmp_path / "source.pt"
    _tiny_model(proprio_dim=source_dim).save_checkpoint(checkpoint)
    model = _tiny_model(proprio_dim=target_dim)
    report = model.load_pretrained_checkpoint(checkpoint, reset_proprio_encoder=True)
    assert report["source_proprio_dim"] == source_dim
    assert report["target_proprio_dim"] == target_dim


@pytest.mark.parametrize("source_path", ["/deleted/original.pt", "${oc.env:FASTWAM_MISSING_PRETRAIN_TEST}", "???"])
def test_training_resume_does_not_read_original_pretrain(tmp_path, monkeypatch, source_path):
    from omegaconf import OmegaConf
    from fastwam import runtime
    monkeypatch.delenv("FASTWAM_MISSING_PRETRAIN_TEST", raising=False)

    cfg = OmegaConf.create({
        "output_dir": str(tmp_path), "resume": "/finetuned/step_000002.pt",
        "seed": 42, "mixed_precision": "no", "data": {"action_dim": 14, "state_dim": 14},
        "model": {
            "_target_": "e05.create_e05",
            "pretrained_checkpoint_path": source_path,
            "vggt_backbone_pretrained_checkpoint_path": source_path,
            "skip_vggt_load_from_pretrain": False,
            "skip_action_dit_load_from_pretrain": False,
            "action_dit_config": {"action_dim": "${data.action_dim}"},
            "proprio_dim": "${data.state_dim}",
        },
    })
    constructed = []

    def instantiate_model(model_cfg, **kwargs):
        assert model_cfg.get("vggt_backbone_pretrained_checkpoint_path") is None
        assert model_cfg.pretrained_checkpoint_path is None
        assert model_cfg.skip_vggt_load_from_pretrain
        assert model_cfg.skip_action_dit_load_from_pretrain
        assert model_cfg.action_dit_config.action_dim == 14
        assert model_cfg.proprio_dim == 14
        constructed.append(True)
        return object()

    class Trainer:
        def __init__(self, **kwargs):
            assert kwargs["cfg"].resume == cfg.resume

        def train(self):
            pass

    monkeypatch.setattr(runtime, "instantiate", instantiate_model)
    monkeypatch.setattr(runtime, "build_datasets", lambda cfg: (None, None))
    monkeypatch.setattr(runtime, "Wan22Trainer", Trainer)
    monkeypatch.setattr(runtime, "_resolve_train_device", lambda: "cpu")
    monkeypatch.setattr(runtime.misc, "register_work_dir", lambda path: None)
    runtime.run_training(cfg)
    assert constructed == [True]
    assert OmegaConf.to_container(cfg.model, resolve=False)["pretrained_checkpoint_path"] == source_path


def test_frozen_vggt_is_saved_in_full_training_state(tmp_path):
    from fastwam.trainer import Wan22Trainer

    source = _tiny_model(action_dim=16, proprio_dim=16)
    checkpoint = tmp_path / "pretrain.pt"
    source.save_checkpoint(checkpoint)
    target = _tiny_model(action_dim=14, proprio_dim=14, train_vggt=False)
    target.load_pretrained_checkpoint(
        checkpoint, reset_action_projections=True, reset_proprio_encoder=True
    )
    Wan22Trainer._apply_dit_only_train_mode(target)
    assert target._modules.get("vggt_backbone") is target.vggt_backbone
    assert all(not p.requires_grad for p in target.vggt_backbone.parameters())
    saved = target.state_dict()
    for key, value in target.vggt_backbone.state_dict().items():
        torch.testing.assert_close(saved[f"vggt_backbone.{key}"], value)
    restored = _tiny_model(action_dim=14, proprio_dim=14, train_vggt=False)
    restored.load_state_dict(saved, strict=True)
    for key, value in target.vggt_backbone.state_dict().items():
        torch.testing.assert_close(restored.vggt_backbone.state_dict()[key], value)
    restored.to(dtype=torch.bfloat16)
    assert restored.vggt_backbone.camera_head.proj.weight.dtype == torch.float32
    assert restored.vggt_backbone.language_projector[0].weight.dtype == torch.bfloat16


def test_inference_missing_checkpoint_fails_before_model_creation(tmp_path, monkeypatch):
    from fastwam import runtime
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "mixed_precision": "no", "model": {"pretrained_checkpoint_path": "/old.pt"},
        "inference": {"device": "cpu", "checkpoint_path": str(tmp_path / "missing.pt")},
    })
    monkeypatch.setattr(runtime, "instantiate", lambda *a, **kw: pytest.fail("Model must not be constructed"))
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        runtime.run_inference(cfg)


def test_full_state_contract_is_saved_and_checked_without_sharing_config(tmp_path):
    import json
    from types import SimpleNamespace
    from fastwam.trainer import Wan22Trainer

    model = _tiny_model(action_dim=14, proprio_dim=14)
    assert not hasattr(model, "sharing_config")
    trainer = object.__new__(Wan22Trainer)
    trainer.model = model
    trainer.global_step, trainer.epoch, trainer.batch_in_epoch = 2, 0, 2
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda x: x, num_processes=8)
    trainer._save_trainer_state(str(tmp_path))
    metadata = json.loads((tmp_path / "trainer_state.json").read_text())
    assert metadata["world_size"] == 8
    assert metadata["model_architecture"] == model.checkpoint_metadata()
    model.validate_training_state_metadata(metadata, 8)

    class ReachedAcceleratorLoad(Exception):
        pass

    def load_state(**kwargs):
        raise ReachedAcceleratorLoad

    trainer.accelerator.load_state = load_state
    with pytest.raises(ReachedAcceleratorLoad):
        trainer.load_training_state(str(tmp_path))
    model.vggt_camera_keys = ("changed_camera",)
    with pytest.raises(ValueError, match="metadata mismatch"):
        trainer.load_training_state(str(tmp_path))


@pytest.mark.parametrize("change", ["camera", "layer_pairs", "action_dim", "proprio_dim", "freeze", "world_size", "legacy"])
def test_full_state_rejects_architecture_or_world_size_changes(change):
    model = _tiny_model(action_dim=14, proprio_dim=14)
    metadata = {"model_architecture": model.checkpoint_metadata(), "world_size": 8}
    architecture = metadata["model_architecture"]
    if change == "camera":
        architecture["vggt_camera_keys"] = ["different"]
    elif change == "layer_pairs":
        architecture["layer_pairs"] = [[1, 0], [0, 1]]
    elif change in {"action_dim", "proprio_dim"}:
        architecture[change] = 16
    elif change == "freeze":
        architecture["train_vggt"] = False
    elif change == "world_size":
        metadata["world_size"] = 4
    else:
        del metadata["model_architecture"]
    with pytest.raises(ValueError, match="full-state resume"):
        model.validate_training_state_metadata(metadata, 8)


@pytest.mark.parametrize("missing", ["both", "camera_only", "layers_only"])
def test_early_weight_checkpoint_without_camera_layer_metadata(tmp_path, missing):
    source = _tiny_model(action_dim=14, proprio_dim=14)
    path = tmp_path / "early_v1.pt"
    source.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    if missing in {"both", "camera_only"}:
        del payload["metadata"]["vggt_camera_keys"]
    if missing in {"both", "layers_only"}:
        del payload["metadata"]["layer_pairs"]
    del payload["metadata"]["reconstruction_target"]
    torch.save(payload, path)
    target = _tiny_model(action_dim=14, proprio_dim=14)
    if missing != "both":
        with pytest.raises(ValueError, match="metadata mismatch"):
            target.load_checkpoint(path)
    else:
        target.load_checkpoint(path)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)


def test_backbone_pretrained_checkpoint_restores_language_and_heads(tmp_path):
    source = _tiny_backbone()
    target = _tiny_backbone()
    checkpoint = tmp_path / "backbone.pt"
    payload = {"format": "beta_future_geometry_backbone_v1",
               "vggt_backbone": source.state_dict(), "metadata": {"step": 30000}}
    torch.save(payload, checkpoint)
    target.load_pretrained_checkpoint(str(checkpoint))
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    del payload["vggt_backbone"]["language_projector.0.weight"]
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="language_projector"):
        target.load_pretrained_checkpoint(str(checkpoint))
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="vggt_backbone"):
        target.load_pretrained_checkpoint(str(checkpoint))


def test_runtime_backbone_pretrain_skips_default_vggt(tmp_path, monkeypatch):
    from e05 import create_e05
    from fastwam.models.wan22 import vggt_lang_recon_backbone

    source = _tiny_backbone()
    checkpoint = tmp_path / "backbone.pt"
    torch.save({"vggt_backbone": source.state_dict()}, checkpoint)

    def build_backbone(**kwargs):
        assert kwargs["skip_vggt_load_from_pretrain"] is True
        return _tiny_backbone()

    monkeypatch.setattr(vggt_lang_recon_backbone, "build_vggt_language_recon_backbone", build_backbone)
    kwargs = dict(
        vggt_weights_path="/missing/vggt.pt",
        vggt_backbone_pretrained_checkpoint_path=str(checkpoint),
        reconstruction_loss_config={"gradient_weight": 0.5, "use_conf_loss": False},
        skip_action_dit_load_from_pretrain=True,
        action_dit_config=dict(hidden_dim=8, action_dim=14, ffn_dim=16, text_dim=10,
                               freq_dim=8, eps=1e-6, num_heads=2, attn_head_dim=4,
                               num_layers=2, use_gradient_checkpointing=False),
        action_scheduler=dict(train_shift=5, infer_shift=5, num_train_timesteps=10),
        layer_pairs=((0, 0), (1, 1)), projection_mode="vggt_to_fastwam",
        vggt_width=64, vggt_num_heads=2, proprio_dim=14,
        device="cpu", model_dtype=torch.float32,
    )
    model = create_e05(**kwargs)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(model.vggt_backbone.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="only one"):
        create_e05(**kwargs, pretrained_checkpoint_path=str(checkpoint))
