"""LIBERO RGB-D conventions shared by preparation and online inference.

New paired data uses upright OpenCV images. The historical LIBERO evaluator's
180-degree observation transform is deliberately left available for old runs.
"""
import numpy as np

CAMERAS = {"image": "agentview", "wrist_image": "robot0_eye_in_hand"}


def to_opencv_image(image, convention):
    if convention not in {"opengl", "opencv"}:
        raise ValueError(f"Unknown robosuite image convention: {convention}")
    return np.ascontiguousarray(image[::-1] if convention == "opengl" else image)


def observation_rgb(obs):
    from robosuite import macros
    return {key: to_opencv_image(obs[f"{camera}_image"], macros.IMAGE_CONVENTION)
            for key, camera in CAMERAS.items()}


def camera_calibration(position, rotation, fovy, height, width):
    """MuJoCo camera pose -> positive-focal K and OpenCV world-to-camera E."""
    focal = 0.5 * height / np.tan(np.deg2rad(fovy) / 2)
    k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    world_to_cv = np.diag([1., -1., -1.]) @ np.asarray(rotation).reshape(3, 3).T
    e = np.column_stack((world_to_cv, -world_to_cv @ np.asarray(position)))
    return k.astype(np.float32), e.astype(np.float32)


def capture_geometry(env, obs):
    from robosuite import macros
    from robosuite.utils.camera_utils import get_real_depth_map
    depths, valid, intrinsics, extrinsics = [], [], [], []
    for camera in CAMERAS.values():
        raw = to_opencv_image(obs[f"{camera}_depth"], macros.IMAGE_CONVENTION)[..., 0]
        depth = get_real_depth_map(env.sim, raw).astype(np.float32)
        mask = np.isfinite(depth) & (depth > 0) & (raw < 1.0 - 1e-6)
        camera_id = env.sim.model.camera_name2id(camera)
        k, e = camera_calibration(env.sim.data.cam_xpos[camera_id],
                                  env.sim.data.cam_xmat[camera_id],
                                  env.sim.model.cam_fovy[camera_id], *depth.shape)
        depths.append(np.where(mask, depth, 0))
        valid.append(mask)
        intrinsics.append(k)
        extrinsics.append(e)
    return dict(depth=np.stack(depths), valid=np.stack(valid),
                intrinsics=np.stack(intrinsics), extrinsics=np.stack(extrinsics))


def project_world(points, k, e):
    camera = np.asarray(points) @ e[:, :3].T + e[:, 3]
    pixels = camera @ k.T
    return pixels[:, :2] / pixels[:, 2:3], camera[:, 2]


def reprojection_report(target):
    """Check independent depth renders across both views (occlusion-aware)."""
    reports = []
    for source, destination in ((0, 1), (1, 0)):
        depth = target['depth'][source]
        h, w = depth.shape
        yy, xx = np.mgrid[8:h:16, 8:w:16]
        uv = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
        z = depth[yy, xx].ravel()
        k, e = target['intrinsics'][source], target['extrinsics'][source]
        camera = (uv @ np.linalg.inv(k).T) * z[:, None]
        world = (camera - e[:, 3]) @ e[:, :3]
        pixels, predicted = project_world(world, target['intrinsics'][destination], target['extrinsics'][destination])
        ix, iy = np.rint(pixels).astype(int).T
        inside = (z > 0) & (predicted > 0) & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        ix, iy, predicted = ix[inside], iy[inside], predicted[inside]
        observed = target['depth'][destination, iy, ix]
        valid = target['valid'][destination, iy, ix]
        # Points behind the visible surface are occluded, not calibration errors.
        visible = valid & (predicted <= observed + .02)
        errors = np.abs(predicted[visible] - observed[visible])
        reports.append({'source': source, 'destination': destination,
                        'visible_points': int(len(errors)),
                        'median_depth_error_m': float(np.median(errors)) if len(errors) else None})
    return reports


def is_noop(action, previous=None, threshold=1e-4):
    return bool(np.linalg.norm(action[:-1]) < threshold and
                (previous is None or action[-1] == previous[-1]))


def encode_action(action):
    result = np.asarray(action, dtype=np.float32).copy()
    result[-1] = (1. - result[-1]) / 2.  # Dataset: 1=open, 0=closed.
    return result
