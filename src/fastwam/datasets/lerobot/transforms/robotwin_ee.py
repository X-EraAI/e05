"""RoboTwin EE poses: [xyz, first two rotation columns, gripper] per arm.

The Rot6D convention in this dataset stores *columns*.  The generic rotation
helpers in this repository store rows, so conversion is kept explicit here.
"""

import torch
import torch.nn.functional as F

from ..utils.rotation import matrix_to_quaternion, quaternion_to_matrix


IDENTITY_ROT6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def columns6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 6:
        raise ValueError(f"Expected six rotation values, got {value.shape}")
    first = F.normalize(value[..., :3], dim=-1)
    second = value[..., 3:] - (first * value[..., 3:]).sum(-1, keepdim=True) * first
    second = F.normalize(second, dim=-1)
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)


def matrix_to_columns6d(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def quaternion_pose_to_ee_state(pose: torch.Tensor) -> torch.Tensor:
    """Convert two [xyz, qwxyz, gripper] arms into the dataset's 20D state."""
    pose = torch.as_tensor(pose)
    if pose.shape[-1] != 16:
        raise ValueError(f"Expected two 8D EE poses, got {pose.shape}")
    arms = pose.reshape(*pose.shape[:-1], 2, 8)
    rotation = quaternion_to_matrix(F.normalize(arms[..., 3:7], dim=-1))
    return torch.cat((arms[..., :3], matrix_to_columns6d(rotation), arms[..., 7:8]), dim=-1).flatten(-2)


def relative_ee_action(target: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Express each absolute target in the fixed local frame of its chunk base."""
    if target.shape[-1] != 20 or base.shape[-1] != 20:
        raise ValueError("EE target and base must each have 20 dimensions")
    target_arm = target.reshape(*target.shape[:-1], 2, 10)
    base_arm = base.reshape(*base.shape[:-1], 2, 10)
    base_rotation = columns6d_to_matrix(base_arm[..., 3:9])
    target_rotation = columns6d_to_matrix(target_arm[..., 3:9])
    local_rotation = base_rotation.transpose(-1, -2) @ target_rotation
    local_position = (
        base_rotation.transpose(-1, -2)
        @ (target_arm[..., :3] - base_arm[..., :3]).unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat((local_position, matrix_to_columns6d(local_rotation), target_arm[..., 9:10]), dim=-1).flatten(-2)


def absolute_ee_action(delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Reconstruct absolute [xyz, qwxyz, gripper] RoboTwin EE commands."""
    if delta.shape[-1] != 20 or base.shape[-1] != 20:
        raise ValueError("EE delta and base must each have 20 dimensions")
    delta_arm = delta.reshape(*delta.shape[:-1], 2, 10)
    base_arm = base.reshape(*base.shape[:-1], 2, 10)
    base_rotation = columns6d_to_matrix(base_arm[..., 3:9])
    local_rotation = columns6d_to_matrix(delta_arm[..., 3:9])
    world_rotation = base_rotation @ local_rotation
    world_position = base_arm[..., :3] + (base_rotation @ delta_arm[..., :3].unsqueeze(-1)).squeeze(-1)
    return torch.cat((world_position, matrix_to_quaternion(world_rotation), delta_arm[..., 9:10]), dim=-1).flatten(-2)


class RoboTwinEEDeltaTransform:
    """Chunk-start-relative EE actions, with absolute EE state and gripper."""

    def __init__(self, key: str = "endpose6d"):
        self.key = key

    def forward(self, batch: dict) -> dict:
        if "action" not in batch:
            return batch
        state = batch["state"][self.key]
        action = batch["action"][self.key]
        # Samples have state [T+1,20]/action [T,20]; episode statistics use
        # state [N,1,20]/action [N,H,20]. Both anchor at the first state.
        base = state[..., :1, :] if state.ndim == 3 else state[:1, :]
        transformed = relative_ee_action(action, base)
        if "action_is_pad" in batch:
            padded = torch.as_tensor(batch["action_is_pad"], dtype=torch.bool, device=action.device)
            if padded.any():
                transformed = transformed.clone()
                transformed[padded] = relative_ee_action(base.expand_as(action), base.expand_as(action))[padded]
        batch["action"][self.key] = transformed
        return batch

    def backward(self, batch: dict) -> dict:
        state = batch["state"][self.key]
        action = batch["action"][self.key]
        base = state[..., :1, :] if state.ndim == 3 else state[:1, :]
        batch["action"][self.key] = absolute_ee_action(action, base)
        return batch
