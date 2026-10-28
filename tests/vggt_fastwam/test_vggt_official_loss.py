"""Regression contracts for the upstream adapter, not reimplementations of its math."""
import torch
import pytest

from fastwam.models.wan22.vggt_official_loss_adapter import official_config, official_batch
from fastwam.models.wan22.vggt_omega_loss.losses.loss import MultitaskLoss
from fastwam.models.wan22.vggt_reconstruction_loss import (
    compute_vggt_reconstruction_loss, compute_legacy_vggt_reconstruction_loss,
    canonical_reconstruction_config,
)


def inputs(views=3):
    b, h, w = 2, 16, 16
    torch.manual_seed(71)
    k = torch.eye(3).repeat(b, views, 1, 1)
    k[..., 0, 0] = k[..., 1, 1] = 16
    k[..., 0, 2] = k[..., 1, 2] = 8
    e = torch.eye(3, 4).repeat(b, views, 1, 1)
    e[..., 0, 3] = torch.arange(views) * .2
    pose = torch.zeros(b, views, 9)
    pose[..., 6] = 1  # xyzw quaternion
    pose[..., 7:] = .9
    return dict(depth=(.8 + torch.rand(b, views, h, w, 1) * .1).requires_grad_(),
                depth_conf=torch.full((b, views, h, w), 1.5, requires_grad=True),
                pose_enc=pose.requires_grad_(), gt_depth=torch.rand(b, views, h, w) + 1,
                gt_valid=torch.ones(b, views, h, w, dtype=torch.bool),
                gt_intrinsics=k, gt_extrinsics=e)


@pytest.mark.parametrize('progress', [0.0, .6, 1.0])
@pytest.mark.parametrize('views', [1, 3])
def test_official_objective_and_gradients_match_upstream(progress, views):
    x = inputs(views)
    gt_copy = {k: v.clone() for k, v in x.items() if k.startswith('gt_')}
    torch.manual_seed(7)
    loss, metrics = compute_vggt_reconstruction_loss(**x, backend='official',
                      schedule_progress=progress, is_synthetic=True)
    leaves = [x[k] for k in ('depth', 'depth_conf', 'pose_enc')]
    grad = torch.autograd.grad(loss, leaves)
    batch = official_batch(**{k: v for k, v in x.items() if k.startswith('gt_')}, is_synthetic=True)
    torch.manual_seed(7)
    direct = MultitaskLoss(**official_config())(
        {'depth': x['depth'], 'depth_conf': x['depth_conf'], 'pose_enc_list': [x['pose_enc']]},
        batch, progress)['loss_objective']
    direct_grad = torch.autograd.grad(direct, leaves)
    torch.testing.assert_close(loss, direct, rtol=0, atol=0)
    for a, b in zip(grad, direct_grad):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(torch.isfinite(v) for v in metrics.values())
    for key in gt_copy:
        assert torch.equal(x[key], gt_copy[key])


def test_point_focal_gradient_switches_at_warmup_boundary():
    from fastwam.models.wan22.vggt_omega_loss.losses.gt_normalization import normalize_gt_batch
    from fastwam.models.wan22.vggt_omega_loss.losses.base_loss.point import compute_point_loss
    x = inputs()
    for progress, expected in [(0., False), (.6, True)]:
        batch = normalize_gt_batch(official_batch(**{k:v for k,v in x.items() if k.startswith('gt_')}), 'global')
        result = compute_point_loss({'depth':x['depth'], 'pose_enc_list':[x['pose_enc']]}, batch,
                                    schedule_progress=progress, intrinsics_warmup_ratio=.6)[2]
        grad = torch.autograd.grad(result['loss_point'], x['pose_enc'])[0]
        assert bool(grad[...,7:].abs().sum() > 0) == expected
        assert grad[...,:3].abs().sum() > 0


def test_legacy_dispatch_exact_and_old_metadata_compatible():
    x = inputs()
    a, ma = compute_legacy_vggt_reconstruction_loss(**x, gradient_weight=.5)
    b, mb = compute_vggt_reconstruction_loss(**x, backend='legacy', official={}, gradient_weight=.5)
    assert torch.equal(a,b)
    assert all(torch.equal(ma[k],mb[k]) for k in ma)
    assert canonical_reconstruction_config({'backend':'legacy', 'official':{}, 'gradient_weight':.5}) == {'gradient_weight':.5}


def test_empty_and_invalid_masks_remain_finite_and_connected():
    for empty in (False, True):
        x = inputs()
        x['gt_valid'].zero_()
        if empty:
            x = {k:v[:0] for k,v in x.items()}
        loss, metrics = compute_vggt_reconstruction_loss(**x, backend='official', schedule_progress=.8)
        assert loss.item() == 0
        grads = torch.autograd.grad(loss, [x[k] for k in ('depth','depth_conf','pose_enc')])
        assert all(torch.isfinite(g).all() for g in grads)
        assert len(metrics) == 5


def test_official_defaults_and_missing_track_labels_are_explicit():
    config = official_config()
    assert config['track'] is None and config['normalize_predictions'] is False
    assert [config[k]['weight'] for k in ('camera','depth','point')] == [5.,5.,5.]
    with pytest.raises(ValueError, match='track labels'):
        official_config({'track': {'weight':5.}})
    with pytest.raises(ValueError, match='schedule_progress'):
        compute_vggt_reconstruction_loss(**inputs(), backend='official')


def test_vendored_sources_only_rewrite_import_paths():
    import hashlib, json
    from pathlib import Path
    from fastwam.models.wan22 import vggt_omega_loss
    root = Path(vggt_omega_loss.__file__).parent
    for name, digest in json.loads((root/'upstream_sha256.json').read_text()).items():
        source = (root/name).read_text().replace('from fastwam.models.wan22.vggt_omega_loss.losses.', 'from losses.').replace('from fastwam.models.wan22.vggt_omega_loss.train_utils.', 'from train_utils.')
        assert hashlib.sha256(source.encode()).hexdigest() == digest, name


def test_normalized_visualization_does_not_report_meter_errors(tmp_path):
    import json
    from experiments.robotwin.fastwam_policy.reconstruction_visualizer import ReconstructionVisualizer
    x = {k:v[:1].detach() for k,v in inputs().items()}
    from fastwam.models.wan22.vggt_omega_loss.losses.gt_normalization import normalize_gt_batch
    batch = normalize_gt_batch(official_batch(**{k:v for k,v in x.items() if k.startswith('gt_')}), 'global')
    reconstruction = {'depth':batch['depths'], 'depth_conf':x['depth_conf'],
                      'pose_enc':x['pose_enc'], 'scale_convention':'scene_normalized'}
    gt = {'depth':x['gt_depth'], 'valid':x['gt_valid'], 'intrinsics':x['gt_intrinsics'], 'extrinsics':x['gt_extrinsics']}
    visualizer = ReconstructionVisualizer(tmp_path, save_video=False)
    frame = visualizer.save(episode_index=0,replan_index=0,vggt_images=torch.rand(1,3,3,16,16),
                            reconstruction=reconstruction, ground_truth=gt)
    metrics = json.loads((frame/'metrics.json').read_text())
    assert 'depth_mae_m' not in metrics
    assert metrics['depth_mae_normalized'] == 0
    assert metrics['current_gt_scene_scale_m'] > 1
    assert 'comment scale_convention scene_normalized' in (frame/'fused_pointcloud.ply').read_text()
    import numpy as np
    with np.load(frame/'reconstruction.npz') as saved:
        assert str(saved['pred_scale_convention']) == 'scene_normalized'
        assert float(saved['gt_scene_scale_m']) == metrics['current_gt_scene_scale_m']


def test_nested_hydra_overrides_merge_without_dropping_official_defaults():
    from omegaconf import OmegaConf
    config = official_config(OmegaConf.create({'depth': {'weight': 2.}}))
    assert config['depth']['weight'] == 2.
    assert config['depth']['use_conf_loss'] is True
    plain = canonical_reconstruction_config(OmegaConf.create({'backend':'official','official':{'depth':{'weight':2.}}}))
    import json
    json.dumps(plain)
