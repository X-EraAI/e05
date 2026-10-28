import pytest
import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.layers.attention import SelfAttention


class TinyPatchEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=2, stride=2)

    def forward(self, images):
        return self.proj(images).flatten(2).transpose(1, 2)


def make_aggregator(callback=None, callback_layer_indices=()):
    model = Aggregator(
        patch_size=2, embed_dim=8, depth=2, num_heads=2, mlp_ratio=2,
        num_register_tokens=1, register_attention_block_indices=[1],
        cached_layer_indices=(0, 1), kv_callback=callback,
        callback_layer_indices=callback_layer_indices,
        patch_embed=TinyPatchEmbed(embed_dim=8),
    )
    for module in model.modules():
        if hasattr(module, "bias_mask"):
            module.bias_mask.fill_(1)
        if hasattr(module, "gamma"):
            nn.init.constant_(module.gamma, 1e-5)
    return model


def test_self_attention_exports_actual_detached_kv():
    torch.manual_seed(0)
    attention = SelfAttention(dim=8, num_heads=2, qkv_bias=True, use_qk_norm=True)
    x = torch.randn(2, 5, 8, requires_grad=True)
    captured = []
    output = attention(x, kv_callback=lambda k, v: captured.append((k, v)))
    qkv = attention.qkv(x).reshape(2, 5, 3, 2, 4)
    _, expected_k, expected_v = torch.unbind(qkv, dim=2)
    expected_k = attention.k_norm(expected_k.transpose(1, 2))
    expected_v = expected_v.transpose(1, 2)
    assert len(captured) == 1
    actual_k, actual_v = captured[0]
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    assert not actual_k.requires_grad and not actual_v.requires_grad
    output.sum().backward()
    assert x.grad is not None


def test_aggregator_callback_covers_selected_global_and_register_layers_only():
    events = []
    model = make_aggregator(events.append, callback_layer_indices=(0, 1)).eval()
    images = torch.randn(1, 2, 3, 4, 4)
    with torch.no_grad():
        outputs, _ = model(images)
    assert [(e['layer_idx'], e['attention_type']) for e in events] == [(0, 'global'), (1, 'register')]
    assert events[0]['k'].shape[-2] == 2 * (1 + 1 + 4)
    assert events[1]['k'].shape[-2] == 2 * (1 + 1)
    assert all(not e['k'].requires_grad and not e['v'].requires_grad for e in events)
    assert len(outputs) == 2


def test_callback_is_default_off_and_does_not_change_outputs_or_state_dict():
    baseline = make_aggregator().eval()
    events = []
    enabled = make_aggregator(events.append, callback_layer_indices=(0, 1)).eval()
    enabled.load_state_dict(baseline.state_dict())
    assert baseline.state_dict().keys() == enabled.state_dict().keys()
    images = torch.randn(1, 2, 3, 4, 4)
    with torch.no_grad():
        expected, _ = baseline(images)
        actual, _ = enabled(images)
    assert all(events)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_callback_payload_mutation_cannot_change_attention_or_gradients():
    torch.manual_seed(1)
    baseline = SelfAttention(dim=8, num_heads=2, qkv_bias=True, use_qk_norm=True)
    enabled = SelfAttention(dim=8, num_heads=2, qkv_bias=True, use_qk_norm=True)
    enabled.load_state_dict(baseline.state_dict())
    x1 = torch.randn(2, 5, 8, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    expected = baseline(x1)
    actual = enabled(x2, kv_callback=lambda k, v: (k.zero_(), v.zero_()))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward(); expected.sum().backward()
    torch.testing.assert_close(x2.grad, x1.grad)


def test_frozen_aggregator_parameters_remain_without_gradients():
    model = make_aggregator(lambda event: None, callback_layer_indices=(0, 1))
    model.requires_grad_(False)
    images = torch.randn(1, 2, 3, 4, 4, requires_grad=True)
    outputs, _ = model(images)
    sum(x.sum() for x in outputs if x is not None).backward()
    assert all(parameter.grad is None for parameter in model.parameters())


def test_callback_with_default_empty_layer_selection_does_not_fire():
    events = []
    model = make_aggregator(events.append).eval()
    with torch.no_grad():
        model(torch.randn(1, 2, 3, 4, 4))
    assert events == []


class ModuleCallback(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.ones(()))
        self.events = []

    def forward(self, event):
        self.events.append(event)


def test_callable_module_callback_does_not_pollute_aggregator_state_dict():
    baseline_keys = set(make_aggregator().state_dict())
    callback = ModuleCallback()

    model = make_aggregator(callback, callback_layer_indices=(0,))

    assert set(model.state_dict()) == baseline_keys
    assert "kv_callback" not in model._modules
    assert model.kv_callback is callback

    with torch.no_grad():
        model(torch.randn(1, 2, 3, 4, 4))
    assert len(callback.events) == 1

    replacement = ModuleCallback()
    model.kv_callback = replacement
    assert model.kv_callback is replacement
    assert "kv_callback" not in model._modules
    assert set(model.state_dict()) == baseline_keys


def test_capture_only_returns_none_and_preserves_default_api():
    events = []
    model = make_aggregator(events.append, callback_layer_indices=(0,)).eval()
    images = torch.randn(1, 2, 3, 4, 4)
    with torch.no_grad():
        assert model(images, capture_only=True) is None
        default = model(images)
    assert isinstance(default, tuple) and len(default) == 2


def test_call_local_callback_and_layers_override_defaults_without_state_pollution():
    default_events, local_events = [], []
    model = make_aggregator(default_events.append, callback_layer_indices=(0,)).eval()
    baseline_keys = set(model.state_dict())
    with torch.no_grad():
        model(torch.randn(1, 2, 3, 4, 4), kv_callback=local_events.append, callback_layer_indices=(1,))
    assert default_events == []
    assert [event["layer_idx"] for event in local_events] == [1]
    assert model.callback_layer_indices == {0}
    assert set(model.state_dict()) == baseline_keys


def test_call_local_callable_module_does_not_enter_state_dict():
    model = make_aggregator().eval(); callback = ModuleCallback()
    baseline = set(model.state_dict())
    with torch.no_grad():
        model(torch.randn(1, 2, 3, 4, 4), kv_callback=callback, callback_layer_indices=(0,))
    assert len(callback.events) == 1
    assert set(model.state_dict()) == baseline
    assert callback not in model.modules()


def test_call_local_callbacks_are_isolated_under_interleaved_threads():
    import threading

    model = make_aggregator().eval()
    barrier = threading.Barrier(2)
    original = model._run_frame_block
    first_calls = set()
    lock = threading.Lock()

    def interleaved(*args, **kwargs):
        thread_id = threading.get_ident()
        with lock:
            first = thread_id not in first_calls
            first_calls.add(thread_id)
        if first:
            barrier.wait(timeout=10)
        return original(*args, **kwargs)

    model._run_frame_block = interleaved
    default_callback = model.kv_callback
    default_layers = model.callback_layer_indices.copy()
    results = {"a": [], "b": []}
    errors = []

    def run(name, value, layer):
        try:
            images = torch.full((1, 2, 3, 4, 4), value)
            with torch.no_grad():
                model(images, capture_only=True, kv_callback=results[name].append, callback_layer_indices=(layer,))
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=("a", 0.0, 0)),
        threading.Thread(target=run, args=("b", 1.0, 1)),
    ]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=20)
    assert not errors and all(not thread.is_alive() for thread in threads)
    assert [event["layer_idx"] for event in results["a"]] == [0]
    assert [event["layer_idx"] for event in results["b"]] == [1]
    assert not torch.equal(results["a"][0]["k"], results["b"][0]["k"])
    assert model.kv_callback is default_callback
    assert model.callback_layer_indices == default_layers


def test_call_local_callback_exception_does_not_pollute_instance_defaults():
    defaults = []
    model = make_aggregator(defaults.append, callback_layer_indices=(0,)).eval()
    baseline = set(model.state_dict())
    def broken(event): raise RuntimeError("callback boom")
    with torch.no_grad(), pytest.raises(RuntimeError, match="callback boom"):
        model(torch.randn(1, 2, 3, 4, 4), kv_callback=broken, callback_layer_indices=(1,))
    assert model.kv_callback is not broken
    assert model.callback_layer_indices == {0}
    assert set(model.state_dict()) == baseline
