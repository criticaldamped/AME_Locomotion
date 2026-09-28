"""Go2-specific contracts. Invalid data is an error, never a synthetic observation."""
import torch
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

JOINTS = [f'{leg}_{joint}_joint' for leg in ('FR', 'FL', 'RR', 'RL') for joint in ('hip', 'thigh', 'calf')]
FEET = [f'{leg}_foot' for leg in ('FR', 'FL', 'RR', 'RL')]

def finite(name, x):
    if not torch.isfinite(x).all():
        raise FloatingPointError(f'{name}: non-finite entries={int((~torch.isfinite(x)).sum())}')
    return x

class LimitedJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids].clone()
        self.lower = (self.limits[..., 0] - self._offset) / self._scale
        self.upper = (self.limits[..., 1] - self._offset) / self._scale
        self.clip_fraction = torch.zeros(self.num_envs, device=self.device)
        assert self._joint_names == JOINTS
        assert ((self.lower < 0) & (self.upper > 0)).all(), 'Default pose outside soft limits'
    def process_actions(self, actions):
        finite('raw actions', actions)
        super().process_actions(actions)
        self.clip_fraction = ((actions < self.lower) | (actions > self.upper)).float().mean(-1)
        self._processed_actions = self._processed_actions.clamp(self.limits[..., 0], self.limits[..., 1])
    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._processed_actions[env_ids] = self._offset[env_ids]
        self.clip_fraction[env_ids] = 0

@configclass
class LimitedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = LimitedJointPositionAction

def elevation_map(env, sensor_cfg):
    sensor = env.scene[sensor_cfg.name]
    hits = finite('height scanner world hits (missed ray)', sensor.data.ray_hits_w)
    robot = env.scene['robot']
    # Origin at BASE, not at the elevated ray start. +x forward, +y left, +z up.
    delta = hits - robot.data.root_pos_w[:, None, :]
    q = yaw_quat(robot.data.root_quat_w)[:, None, :].expand(-1, hits.shape[1], -1)
    points = quat_apply_inverse(q.reshape(-1, 4), delta.reshape(-1, 3)).reshape_as(hits)
    finite('base yaw map', points)
    # Preserve real step heights; no silent NaN replacement or z=0 ceiling.
    return points.flatten(1)

def stand_still(env):
    robot = env.scene['robot']
    cmd = env.command_manager.get_command('base_velocity')
    standing = torch.linalg.vector_norm(cmd, dim=-1) < 0.05
    return torch.sum(torch.abs(robot.data.joint_pos - robot.data.default_joint_pos), -1) * standing

def saturation_excess(env):
    a = env.action_manager.get_term('joint_pos')
    return ((a.raw_actions - a.raw_actions.clamp(a.lower, a.upper)) ** 2).sum(-1)

def feet_slide(env, sensor_cfg, asset_cfg):
    contact = env.scene[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).amax(1) > 1
    velocity = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return (contact * velocity).sum(-1)

def terrain_progress(env, env_ids):
    terrain = env.scene.terrain
    robot = env.scene['robot']
    # Rough training has +x commands, zero yaw, deterministic reset heading.
    distance = robot.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    elapsed = env.episode_length_buf[env_ids] * env.step_dt
    cmd = env.command_manager.get_command('base_velocity')[env_ids, 0]
    moving = cmd > 0.1
    alive = ~env.termination_manager.terminated[env_ids]
    up = moving & alive & (elapsed > 5.0) & (distance > 0.65 * cmd * elapsed) & (distance > 2.0)
    down = moving & (elapsed > 1.0) & ((distance < 0.3 * cmd * elapsed) | ~alive)
    terrain.terrain_levels[env_ids] = (terrain.terrain_levels[env_ids] + up.long() - (down & ~up).long()).clamp(0, terrain.max_terrain_level - 1)
    terrain.env_origins[env_ids] = terrain.terrain_origins[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
    return terrain.terrain_levels.float().mean()

def track_linear(env, std):
    cmd = env.command_manager.get_command('base_velocity')[:, :2]
    velocity = env.scene['robot'].data.root_lin_vel_b[:, :2]
    score = torch.exp(-((cmd - velocity)**2).sum(-1) / std**2)
    moving = cmd.norm(dim=-1) > 0.05
    # A stationary robot receives zero translational score for a moving command.
    baseline = torch.exp(-(cmd**2).sum(-1) / std**2)
    return score - moving * baseline

def track_yaw(env, std):
    cmd = env.command_manager.get_command('base_velocity')
    robot = env.scene['robot']
    score = torch.exp(-(cmd[:, 2] - robot.data.root_ang_vel_b[:, 2])**2 / std**2)
    speed_sq = (cmd[:, :2]**2).sum(-1)
    progress = (robot.data.root_lin_vel_b[:, :2] * cmd[:, :2]).sum(-1) / speed_sq.clamp_min(1e-6)
    # Do not pay full zero-yaw reward for refusing a translation command.
    gate = torch.where(speed_sq > 0.05**2, progress.clamp(0., 1.), 1.)
    return score * gate
