from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def ray_hits_s(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Get the 3D coordinates of scan points in the sensor's local coordinate 
    frame.

    This function converts the ray hit points from world coordinates to
    sensor-local coordinates. The returned coordinates represent the relative
    positions of scan points w.r.t. the sensor frame.

    Args:
        env: The environment containing the sensor.
        sensor_cfg: The SceneEntity configuration for the sensor.

    Returns:
        A flattened tensor containing the 3D coordinates of scan points in 
        sensor frame. The original shape (N, B, 3) is flattened to (N*B*3,) 
        for compatibility with other observation terms.
    """
    
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w  # (N, 4)
    N, B, _ = relative_pos_w.shape
   
    # Handle different sensor alignments
    alignment = getattr(sensor.cfg, "ray_alignment", "base")
    if alignment == "yaw":
        sensor_quat = yaw_quat(sensor_quat)

    sensor_quat_expanded = (sensor_quat.unsqueeze(1).expand(N, B, 4).reshape(N*B, 4)).to(torch.float)
    relative_pos_w_reshaped = relative_pos_w.reshape(N*B, 3)
    sensor_coords = quat_apply_inverse(sensor_quat_expanded, relative_pos_w_reshaped)
    sensor_coords = sensor_coords.reshape(N, B, 3)

    # print("ray_hits_s:", torch.round(sensor_coords * 100) / 100)
    # print(sensor_coords.reshape(N, B*3))

    if torch.isnan(sensor_coords).any() or torch.isinf(sensor_coords).any():
        print(f"Warning: ray_hits_s contains NaN or Inf: {sensor_coords}")
        sensor_coords = torch.nan_to_num(sensor_coords)
        
    return sensor_coords.reshape(N, B*3)


def elevation_map(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, noise: bool = False) -> torch.Tensor:
    
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w  # (N, 4)
    N, B, _ = relative_pos_w.shape
   
    # Handle different sensor alignments
    alignment = getattr(sensor.cfg, "ray_alignment", "base")
    if alignment == "yaw":
        sensor_quat = yaw_quat(sensor_quat)

    sensor_quat_expanded = (sensor_quat.unsqueeze(1).expand(N, B, 4).reshape(N*B, 4)).to(torch.float)
    relative_pos_w_reshaped = relative_pos_w.reshape(N*B, 3)
    sensor_coords = quat_apply_inverse(sensor_quat_expanded, relative_pos_w_reshaped)
    sensor_coords = sensor_coords.reshape(N, B, 3)

    if torch.isnan(sensor_coords).any() or torch.isinf(sensor_coords).any():
        # print(f"Warning: elevation_map contains NaN or Inf: {sensor_coords}")
        sensor_coords = torch.nan_to_num(sensor_coords)

    if noise:
        # Initialize buffers for shift and delayed observation
        # if getattr(env, "_elevation_map_shift", None) is None:
        #     env._elevation_map_shift = torch.zeros((env.num_envs, 2), device=env.device)
        if getattr(env, "_elevation_map_offset", None) is None or env._elevation_map_offset.shape != (N, 1):
            env._elevation_map_offset = torch.zeros((N, 1), device=env.device)
        # if getattr(env, "_last_elevation_map", None) is None or env._last_elevation_map.shape != (N, B * 3):
        #     env._last_elevation_map = torch.zeros((N, B * 3), device=env.device)

        # Resample x and y shift for reset environments (-3cm to 3cm)
        if hasattr(env, "reset_buf"):
            reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                # env._elevation_map_shift[reset_env_ids] = torch.rand((len(reset_env_ids), 2), device=env.device) * 0.06 - 0.03
                env._elevation_map_offset[reset_env_ids] = torch.rand((len(reset_env_ids), 1), device=env.device) * 0.1 - 0.05

        # Apply x, y shift
        # sensor_coords[..., :2] += env._elevation_map_shift.unsqueeze(1)

        # Add Gaussian noise to each height value
        height_noise = torch.randn_like(sensor_coords[..., 2]) * 0.03 # std dev of 3 cm
        sensor_coords[..., 2] += height_noise  
        # Add a global offset noise to simulate sensor initialization error
        offset_noise = env._elevation_map_offset
        sensor_coords[..., 2] += offset_noise
        
    # Clip height values
    sensor_coords[..., 2] = torch.clamp(sensor_coords[..., 2], min=-1.2, max=0.0)

    current_map = sensor_coords.reshape(N, B * 3)

    return current_map
    
    # if noise:
    #     # Apply data delay: update map only every 5 steps or on reset (10Hz instead of 50Hz)
    #     update_mask = (env.episode_length_buf % 5 == 0)
    #     if hasattr(env, "reset_buf"):
    #         update_mask = update_mask | env.reset_buf
    #     update_env_ids = update_mask.nonzero(as_tuple=False).squeeze(-1)
        
    #     if len(update_env_ids) > 0:
    #         env._last_elevation_map[update_env_ids] = current_map[update_env_ids]

    #     return env._last_elevation_map.clone()
    # else:
    #     return current_map