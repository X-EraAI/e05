"""Joint RGB/calibration augmentation using pinned, unmodified Omega helpers.

Unlike upstream reconstruction, FastWAM predicts future geometry from current
RGB: each camera's geometric parameters are replayed at both time indices.
"""
from contextlib import contextmanager
import hashlib
import random

import numpy as np
from omegaconf import OmegaConf
import torch

from .vggt_omega_official.augmentation import get_image_augmentation
from .vggt_omega_official.base_dataset import BaseDataset

UPSTREAM_COMMIT = "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6"
ASPECT_RATIOS = (5/9, 10/16, 2/3, 3/4, 4/5, 1., 5/4, 4/3, 3/2, 16/10, 9/5)


def augmentation_seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:4], 'big')


@contextmanager
def isolated_rng(seed):
    """Upstream uses three CPU RNGs. Isolate them from all other data processing."""
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


class OfficialVGGTAugmentation:
    def __init__(self, config):
        self.config = OmegaConf.create(OmegaConf.to_container(config, resolve=True)
                                       if OmegaConf.is_config(config) else config)
        self.seed = int(self.config.get('seed', 42))
        self.geometry = BaseDataset(self.config)
        self.geometry.training = True
        self.geometry.edge_rtol_thres = 0.0
        aug = self.config.augs
        if not aug.use_discrete_aspect_ratios:
            raise ValueError('Official FastWAM augmentation requires discrete aspect ratios.')
        self.aspects = tuple(a for a in ASPECT_RATIOS if aug.aspects[0] <= a <= aug.aspects[1])
        if not self.aspects:
            raise ValueError('No aspect ratios within the configured range.')
        for name, value in [('same_preprocess_ratio', self.config.same_preprocess_ratio),
                            ('random_rotation_ratio', self.config.random_rotation_ratio),
                            ('cojitter_ratio', aug.cojitter_ratio)]:
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0, 1].')
        self.image_aug = get_image_augmentation(
            color_jitter=dict(aug.color_jitter), gray_scale=aug.gray_scale,
            gau_blur=aug.gau_blur, jpeg_compression=aug.jpeg_compression,
            low_res_blur=aug.low_res_blur, low_res_blur_min_ratio=aug.low_res_blur_min_ratio,
        ) if aug.enable_image_aug else None

    def contract(self):
        return {'upstream_commit': UPSTREAM_COMMIT, 'adapter_version': 1,
                'config': OmegaConf.to_container(self.config, resolve=True)}

    def __call__(self, images, targets, *, sample_seed, batch_seed):
        aspect = random.Random(batch_seed).choice(self.aspects)
        shape = self.geometry.get_target_shape(aspect)
        if 'current' not in targets:
            raise ValueError('Joint augmentation requires current calibration.')
        views = len(images)
        for name, target in targets.items():
            d, valid, k, e = (target[key] for key in ('depth', 'valid', 'intrinsics', 'extrinsics'))
            if d.ndim != 3 or d.shape[0] != views or valid.shape != d.shape:
                raise ValueError(f'{name}: invalid depth/mask shape.')
            if k.shape != (views, 3, 3) or e.shape != (views, 3, 4):
                raise ValueError(f'{name}: invalid calibration shape.')
            if not torch.isfinite(k).all() or not torch.isfinite(e).all() or (k[:, (0, 1), (0, 1)] <= 0).any():
                raise ValueError(f'{name}: invalid calibration values.')
            if (valid & (~torch.isfinite(d) | (d <= 0))).any():
                raise ValueError(f'{name}: invalid metric depth in valid pixels.')
            for v, image in enumerate(images):
                if image.ndim != 3 or image.shape[0] != 3 or image.shape[-2:] != d[v].shape:
                    raise ValueError(f'{name}: native RGB/depth shapes disagree.')

        with isolated_rng(sample_seed):
            shared = random.random() < self.config.same_preprocess_ratio
            common = self.geometry.sample_preprocess_params() if shared else None
            params = [common if shared else self.geometry.sample_preprocess_params() for _ in images]
            output, rgb_output = {}, []
            for name, target in targets.items():
                result = {key: [] for key in ('depth', 'valid', 'intrinsics', 'extrinsics')}
                for v, rgb in enumerate(images):
                    if rgb.dtype == torch.uint8:
                        rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
                    else:
                        if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                            raise ValueError('Float RGB must be finite and in [0, 1].')
                        rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                    depth = target['depth'][v].float().cpu().numpy().copy()
                    depth[~target['valid'][v].cpu().numpy().astype(bool)] = 0
                    image, depth, extrinsics, intrinsics, _ = self.geometry.process_one_image(
                        rgb_np, depth, target['extrinsics'][v].float().cpu().numpy(),
                        target['intrinsics'][v].float().cpu().numpy(), shape,
                        preprocess_params=params[v],
                    )
                    if tuple(image.shape[:2]) != tuple(shape):
                        raise ValueError('Official augmentation produced an inconsistent shape.')
                    if name == 'current':
                        rgb_output.append(torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255)
                    for key, value in [('depth', depth), ('valid', np.isfinite(depth) & (depth > 0)),
                                       ('intrinsics', intrinsics), ('extrinsics', extrinsics)]:
                        result[key].append(torch.from_numpy(np.ascontiguousarray(value)))
                output[name] = {key: torch.stack(value) for key, value in result.items()}
            rgb_output = torch.stack(rgb_output)
            shared_photo = False
            if self.image_aug is not None:
                shared_photo = bool(self.config.augs.cojitter and random.random() > self.config.augs.cojitter_ratio)
                if shared_photo:
                    rgb_output = self.image_aug(rgb_output)
                else:
                    rgb_output = torch.stack([self.image_aug(rgb) for rgb in rgb_output])
            # Bicubic low-res simulation can overshoot, as in upstream; preserve parity.
            trace = torch.tensor([sample_seed, batch_seed, self.aspects.index(aspect), int(shared),
                                  int(shared_photo), *[p['rot_aug_angle'] or 0 for p in params]], dtype=torch.int64)
        return rgb_output, output, trace
