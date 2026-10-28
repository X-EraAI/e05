"""Current/future target semantics, cache independence and checkpoint contracts."""
from unittest.mock import patch

import pytest
import torch

from test_e05 import _tiny_model, _future_sample, _prediction


def current_sample(batch=2):
    return {key.replace('vggt_recon_future_', 'vggt_recon_current_'): value
            for key, value in _future_sample(batch).items()}


@pytest.mark.parametrize('backend', ['legacy', 'official'])
def test_current_training_ignores_future_labels_and_padding_and_has_gradients(backend):
    model = _tiny_model(reconstruction_target='current')
    model.reconstruction_loss_config['backend'] = backend
    batch = current_sample()
    # Official depth supervision requires more valid pixels than the tiny 4x4 fixture.
    batch['vggt_images'] = torch.rand(2, 2, 3, 16, 16)
    batch['vggt_recon_current_depth'] = torch.ones(2, 2, 16, 16)
    batch['vggt_recon_current_depth_valid'] = torch.ones(2, 2, 16, 16, dtype=torch.bool)
    torch.manual_seed(8)
    expected, metrics = model.training_loss(batch)
    # Poison future supervision, including an all-padding batch. Current must
    # produce the identical loss, not skip these perfectly valid observations.
    batch.update({key: torch.full_like(value, float('nan'))
                  for key, value in _future_sample().items()
                  if key.startswith('vggt_recon_future_') and value.is_floating_point()})
    batch['vggt_recon_future_is_pad'] = torch.ones(2, dtype=torch.bool)
    batch['vggt_recon_current_depth_valid'][0, 0, 0, 0] = False
    # Check the padding/label independence with the same current valid mask.
    torch.manual_seed(8)
    actual, actual_metrics = model.training_loss(batch)
    baseline = {k: v for k, v in batch.items() if not k.startswith('vggt_recon_future_')}
    torch.manual_seed(8)
    reference, reference_metrics = model.training_loss(baseline)
    torch.testing.assert_close(actual, reference)
    assert actual_metrics.keys() == reference_metrics.keys()
    assert 'loss_vggt_current_depth' in metrics and 'loss_vggt_future_depth' not in metrics
    assert actual_metrics['vggt_current_valid_ratio'] == 1
    assert torch.isfinite(expected) and torch.isfinite(actual)
    actual.backward()
    b = model.vggt_backbone
    for module in (b.aggregator, b.language_projector, b.camera_head, b.dense_head):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)


@pytest.mark.parametrize('target', ['current', 'future'])
def test_loss_selects_configured_supervision(target):
    model = _tiny_model(reconstruction_target=target)
    sample = {**current_sample(), **_future_sample()}
    sample['vggt_recon_current_depth'].fill_(2)
    sample['vggt_recon_future_depth'].fill_(5)
    with patch('fastwam.models.wan22.e05.compute_vggt_reconstruction_loss',
               return_value=(torch.tensor(0.), {})) as loss:
        model._target_recon_loss(_prediction(), sample)
    assert torch.equal(loss.call_args.kwargs['gt_depth'], sample[f'vggt_recon_{target}_depth'])
    assert torch.equal(loss.call_args.kwargs['gt_valid'], sample[f'vggt_recon_{target}_depth_valid'])


@pytest.mark.parametrize('target', ['current', 'future'])
def test_checkpoint_and_inference_target_roundtrip(target, tmp_path):
    model = _tiny_model(reconstruction_target=target)
    path = tmp_path / 'weights.pt'
    model.save_checkpoint(path, step=2)
    restored = _tiny_model(reconstruction_target=target)
    payload = restored.load_checkpoint(path)
    assert payload['target'] == target
    sample = current_sample(1)
    kwargs = dict(action_horizon=4, vggt_images=sample['vggt_images'],
                  context=sample['context'], context_mask=sample['context_mask'],
                  language_context_mask=sample['language_context_mask'],
                  return_reconstruction=True, num_inference_steps=1, seed=7)
    a, b = model.infer_action(**kwargs), restored.infer_action(**kwargs)
    assert set(b['reconstruction']) == {target}
    for key, shape in [('depth', (1, 2, 4, 4)), ('confidence', (1, 2, 4, 4)), ('pose', (1, 2, 9))]:
        value = b['reconstruction'][target][key]
        assert value.shape == shape and torch.isfinite(value).all()
        torch.testing.assert_close(a['reconstruction'][target][key], value)
    torch.testing.assert_close(a['action'], b['action'])


def test_legacy_future_strict_load_and_explicit_current_transfer(tmp_path):
    future = _tiny_model()
    path = tmp_path / 'old.pt'
    future.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    del payload['metadata']['reconstruction_target']
    torch.save(payload, path)
    _tiny_model().load_checkpoint(path)
    current = _tiny_model(reconstruction_target='current')
    with pytest.raises(ValueError, match='target mismatch'):
        current.load_checkpoint(path)
    report = current.load_pretrained_checkpoint(path)
    assert report['metadata_overrides']['reconstruction_target'] == {'source': 'future', 'target': 'current'}
    for key, value in future.state_dict().items():
        torch.testing.assert_close(value, current.state_dict()[key])
    old_state = future.checkpoint_metadata()
    del old_state['reconstruction_target']
    future.validate_training_state_metadata({'model_architecture': old_state, 'world_size': 8}, 8)
    with pytest.raises(ValueError, match='metadata mismatch'):
        current.validate_training_state_metadata({'model_architecture': old_state, 'world_size': 8}, 8)
    current.validate_training_state_metadata({'model_architecture': current.checkpoint_metadata(), 'world_size': 8}, 8)


def test_invalid_target_rejected_before_factory_model_allocation():
    from e05 import create_e05
    with pytest.raises(ValueError, match='reconstruction_target'):
        create_e05(reconstruction_target='both',
            vggt_weights_path='', layer_pairs=[], projection_mode='',
            action_dit_config={}, action_scheduler={})
    with pytest.raises(ValueError, match='reconstruction_target'):
        _tiny_model(reconstruction_target='both')


@pytest.mark.parametrize('index', [0, 4])
def test_current_dataset_reads_only_current_cache_including_episode_end(tmp_path, index):
    from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
    from fastwam.datasets.lerobot.vggt_reconstruction_cache import VGGTReconstructionTargetCache
    # Only the requested current entry exists; the future entry is absent.
    torch.save({'depth': torch.full((2, 4, 4), 3.),
                'intrinsics': torch.eye(3).repeat(2, 1, 1),
                'extrinsics': torch.eye(4)[:3].repeat(2, 1, 1)}, tmp_path / f'{index:08d}.pt')
    sample = dict(idx=index, pixel_values=torch.zeros(2, 5, 3, 4, 4),
                  vggt_images=torch.zeros(2, 3, 4, 4), action=torch.zeros(4, 2),
                  proprio=torch.zeros(5, 2), instruction='move',
                  image_is_pad=torch.tensor([False, True, True, True, True]),
                  action_is_pad=torch.ones(4, dtype=torch.bool),
                  proprio_is_pad=torch.ones(5, dtype=torch.bool))
    class Frames:
        episode_data_index = {'to': torch.tensor([5])}
        def __getitem__(self, _):
            return dict(sample)
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.lerobot_dataset = Frames()
    dataset.reconstruction_target = 'current'
    dataset.reconstruction_target_cache = VGGTReconstructionTargetCache(tmp_path)
    dataset.reconstruction_future_frame_index = 3
    dataset.global_sample_stride = 1
    dataset.num_frames = 5
    dataset.max_padding_retry = 0
    dataset.skip_padding_as_possible = False
    dataset.video_sample_indices = list(range(5))
    dataset.concat_multi_camera = 'horizontal'
    dataset.override_instruction = None
    dataset.resize_transform = dataset.crop_transform = dataset.normalize_transform = lambda x: x
    dataset._get_cached_text_context = lambda _: (torch.zeros(3, 10), torch.ones(3, dtype=torch.bool))
    with patch.object(dataset.reconstruction_target_cache, 'get_pair', side_effect=AssertionError('Future read')), \
         patch.object(dataset.reconstruction_target_cache, 'get', wraps=dataset.reconstruction_target_cache.get) as get:
        output = dataset._get(0)
    get.assert_called_once_with(index)
    assert not any(k.startswith('vggt_recon_future_') for k in output)
    assert torch.all(output['vggt_recon_current_depth'] == 3)
    assert output['vggt_recon_current_depth_valid'].all()




