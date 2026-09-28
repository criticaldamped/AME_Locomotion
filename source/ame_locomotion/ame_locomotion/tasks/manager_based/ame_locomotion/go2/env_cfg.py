"""Independent Go2 AME tasks, using Isaac Lab's quadruped baseline settings."""
from pathlib import Path
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, CurriculumTermCfg as CurrTerm, SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Noise
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as lab_mdp
from . import mdp

ROOT = Path(__file__).resolve().parents[7]
JOINT_CFG = SceneEntityCfg('robot', joint_names=mdp.JOINTS, preserve_order=True)

@configclass
class PolicyObs(ObsGroup):
    angular_velocity = ObsTerm(func=lab_mdp.base_ang_vel, scale=0.2, noise=Noise(n_min=-0.1,n_max=0.1))
    gravity = ObsTerm(func=lab_mdp.projected_gravity, noise=Noise(n_min=-0.03,n_max=0.03))
    command = ObsTerm(func=lab_mdp.generated_commands, params={'command_name':'base_velocity'})
    joint_pos = ObsTerm(func=lab_mdp.joint_pos_rel, params={'asset_cfg':JOINT_CFG}, noise=Noise(n_min=-0.01,n_max=0.01))
    joint_vel = ObsTerm(func=lab_mdp.joint_vel_rel, params={'asset_cfg':JOINT_CFG}, scale=0.05, noise=Noise(n_min=-1.0,n_max=1.0))
    # Raw latent Gaussian action, same quantity as PPO storage and action_rate penalty.
    actions = ObsTerm(func=lab_mdp.last_action)
    height_map = ObsTerm(func=mdp.elevation_map, params={'sensor_cfg':SceneEntityCfg('height_scanner')})
    def __post_init__(self):
        self.concatenate_terms=True
        self.enable_corruption=False

@configclass
class CriticObs(ObsGroup):
    linear_velocity = ObsTerm(func=lab_mdp.base_lin_vel)
    angular_velocity = ObsTerm(func=lab_mdp.base_ang_vel, scale=0.2)
    gravity = ObsTerm(func=lab_mdp.projected_gravity)
    command = ObsTerm(func=lab_mdp.generated_commands, params={'command_name':'base_velocity'})
    joint_pos = ObsTerm(func=lab_mdp.joint_pos_rel, params={'asset_cfg':JOINT_CFG})
    joint_vel = ObsTerm(func=lab_mdp.joint_vel_rel, params={'asset_cfg':JOINT_CFG}, scale=0.05)
    actions = ObsTerm(func=lab_mdp.last_action)
    height_map = ObsTerm(func=mdp.elevation_map, params={'sensor_cfg':SceneEntityCfg('height_scanner')})
    def __post_init__(self):
        self.concatenate_terms=True
        self.enable_corruption=False

@configclass
class Observations:
    policy: PolicyObs = PolicyObs()
    critic: CriticObs = CriticObs()

@configclass
class Go2BaseEnvCfg(UnitreeGo2RoughEnvCfg):
    observations: Observations = Observations()
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=512
        self.scene.robot.spawn.usd_path=str(ROOT/'unitree_model/Go2/usd/go2.usd')
        # Explicit DC torque-speed envelope: Isaac Lab 2.3.2 baseline, 23.5 Nm / 30 rad/s.
        motor=self.scene.robot.actuators['base_legs']
        motor.effort_limit_sim=23.5
        motor.velocity_limit_sim=30.0
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count=1
        self.scene.terrain.terrain_type='plane'
        self.scene.terrain.terrain_generator=None
        self.scene.terrain.visual_material=None
        self.scene.sky_light.spawn=sim_utils.DomeLightCfg(intensity=750.0)
        self.scene.height_scanner.pattern_cfg=patterns.GridPatternCfg(resolution=0.05,size=(1.6,1.0),ordering='xy')
        self.scene.height_scanner.offset.pos=(0.,0.,2.)
        self.scene.height_scanner.ray_alignment='yaw'
        self.curriculum.terrain_levels=None
        self.actions.joint_pos=mdp.LimitedJointPositionActionCfg(asset_name='robot',joint_names=mdp.JOINTS,preserve_order=True,scale=0.25,use_default_offset=True)
        c=self.commands.base_velocity
        c.debug_vis=False
        c.heading_command=False
        c.rel_heading_envs=0.0
        c.rel_standing_envs=0.1
        c.resampling_time_range=(20.,20.)
        c.ranges.lin_vel_x=(0.2,1.0)
        c.ranges.lin_vel_y=(-0.2,0.2)
        c.ranges.ang_vel_z=(-0.5,0.5)
        self.events.add_base_mass=None
        self.events.base_com=None
        self.events.base_external_force_torque=None
        self.events.push_robot=None
        self.events.reset_robot_joints.params['velocity_range']=(0.,0.)
        self.events.reset_base.params={'pose_range':{'yaw':(-3.14,3.14)},'velocity_range':{}}
        r=self.rewards
        r.track_lin_vel_xy_exp=RewTerm(func=mdp.track_linear,weight=2.0,params={"std":0.35})
        r.track_ang_vel_z_exp=RewTerm(func=mdp.track_yaw,weight=0.5,params={"std":0.5})
        r.feet_air_time.weight=0.1
        r.feet_air_time.params['threshold']=0.3
        r.dof_torques_l2.weight=-2e-4
        r.flat_orientation_l2.weight=-0.5
        r.dof_pos_limits.weight=-1.0
        r.undesired_contacts=RewTerm(func=lab_mdp.undesired_contacts,weight=-1.,params={'threshold':1.,'sensor_cfg':SceneEntityCfg('contact_forces',body_names=['.*_thigh','.*_calf'])})
        r.feet_slide=RewTerm(func=mdp.feet_slide,weight=-0.1,params={'sensor_cfg':SceneEntityCfg('contact_forces',body_names=mdp.FEET,preserve_order=True),'asset_cfg':SceneEntityCfg('robot',body_names=mdp.FEET,preserve_order=True)})
        r.stand_still=RewTerm(func=mdp.stand_still,weight=-0.2)
        r.saturation=RewTerm(func=mdp.saturation_excess,weight=-0.05)
        r.termination=RewTerm(func=lab_mdp.is_terminated,weight=-5.0)
        self.terminations.bad_orientation=DoneTerm(func=lab_mdp.bad_orientation,params={'limit_angle':1.0})

@configclass
class Go2RoughEnvCfg(Go2BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type='generator'
        self.scene.terrain.max_init_terrain_level=0
        self.scene.terrain.terrain_generator=terrain_gen.TerrainGeneratorCfg(seed=42,size=(8.,8.),border_width=10.,num_rows=6,num_cols=6,horizontal_scale=0.1,vertical_scale=0.005,slope_threshold=0.75,use_cache=False,curriculum=True,sub_terrains={
            'stairs':terrain_gen.MeshPyramidStairsTerrainCfg(proportion=0.35,step_height_range=(0.02,0.14),step_width=0.35,platform_width=3.,border_width=1.,holes=False),
            'stairs_down':terrain_gen.MeshInvertedPyramidStairsTerrainCfg(proportion=0.35,step_height_range=(0.02,0.14),step_width=0.35,platform_width=3.,border_width=1.,holes=False),
            'rough':terrain_gen.HfRandomUniformTerrainCfg(proportion=0.3,noise_range=(0.01,0.04),noise_step=0.01,border_width=0.25)})
        self.curriculum.terrain_levels=CurrTerm(func=mdp.terrain_progress)
        self.commands.base_velocity.ranges.lin_vel_x=(0.3,0.8)
        self.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
        self.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
        self.events.reset_base.params={'pose_range':{},'velocity_range':{}}
        self.observations.policy.enable_corruption=True
        self.events.physics_material.params.update(static_friction_range=(0.6,1.0),dynamic_friction_range=(0.5,0.6))


def evaluation(cfg):
    cfg.scene.num_envs=16
    cfg.observations.policy.enable_corruption=False
    cfg.events.reset_base.params={'pose_range':{},'velocity_range':{}}
    cfg.events.physics_material.params.update(static_friction_range=(0.8,0.8),dynamic_friction_range=(0.6,0.6))
    cfg.commands.base_velocity.rel_standing_envs=0.
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.5,0.5)
    cfg.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
    cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
    cfg.curriculum.terrain_levels=None

@configclass
class Go2FlatPlayEnvCfg(Go2BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        evaluation(self)

@configclass
class Go2ObstaclePlayEnvCfg(Go2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        evaluation(self)
        g=self.scene.terrain.terrain_generator
        g.num_rows=4
        g.num_cols=4
        # Keep ordered difficulty rows; disable only progression, not generation ordering.
        g.curriculum=True
        g.sub_terrains={'stairs':terrain_gen.MeshPyramidStairsTerrainCfg(proportion=1.,step_height_range=(0.02,0.10),step_width=0.35,platform_width=1.2,border_width=1.,holes=False)}
