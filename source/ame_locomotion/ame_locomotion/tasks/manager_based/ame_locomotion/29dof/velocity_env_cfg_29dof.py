# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ame_locomotion.tasks.manager_based.ame_locomotion import mdp
# import isaaclab.terrains as terrain_gen
import ame_locomotion.tasks.manager_based.ame_locomotion.terrains as terrain_gen

FINETUNE = False

##
# Pre-defined configs
##
from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.terrain_cfg import ROUGH_TERRAINS_CFG  # isort: skip
from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.finetune_terrain_cfg import FINETUNE_ROUGH_TERRAINS_CFG
from ame_locomotion.tasks.manager_based.ame_locomotion.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FINETUNE_ROUGH_TERRAINS_CFG if FINETUNE else ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0]),  # 0.05m resolution, 1.6m x 1.0m, grid 33x21
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),   # 0.1m resolution, 1.6m x 1.0m, grid 17x11
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-2.0, n_max=2.0))    # 1.5
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": True},
        )

        def __post_init__(self):
            self.enable_corruption = True   # Enable observation noise (default False)
            self.concatenate_terms = True   # Concatenate observation terms (default True)

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
        )

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),    # 0.8
            "dynamic_friction_range": (0.3, 1.0),   # 0.6
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),  # "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25}
    )

    # -- penalties
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2, 
        weight=-0.05
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]), # All body parts except feet
        },
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-1.5e-7,
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-0.001
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
    )
    dof_torques_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.01,
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    # -- style
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.6,
        },
    )
    feet_air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )    
    
    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-1.0,
        params={
            "threshold": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"), 
        },
    )
    # -- coordination (cross-body coordination)
    joint_coordination = RewTerm(
        func=mdp.joint_coordination_rel,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "coord_joints": [
                # Cross-side coordination: left leg forward swing with right arm forward swing
                ["left_hip_pitch_joint", "right_shoulder_pitch_joint"],
                ["right_hip_pitch_joint", "left_shoulder_pitch_joint"],
            ],
            "coord_signs": [
                [1.0, 1.0],  # Left hip and right shoulder move in the same direction
                [1.0, 1.0],  # Right hip and left shoulder move in the same direction
            ],
        },
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                body_names=[
                    "torso_link",
                    ".*_shoulder_.*_link",
                    ".*_hip_.*_link",
                    ".*_knee_link",
                    ".*_elbow_link",
                    "waist_.*_link",
                    "pelvis",
                ]
                # body_names=["torso_link"]
            ), "threshold": 1.0},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class MyViewerCfg(ViewerCfg):
    """Configuration for the viewer."""
    eye = (0.0, 0.0, 10.0)
    lookat = (0.0, 0.0, 0.0)

##
# Environment configuration
##


@configclass
class G1RoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # viewer: MyViewerCfg = MyViewerCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        if FINETUNE:
            self.events.reset_base.params= {
                "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                }
            }
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)
            
            self.rewards.termination_penalty.weight = -200.0
            self.rewards.track_lin_vel_xy_exp.weight = 2.0
            self.rewards.track_ang_vel_z_exp.weight = 3.0
            self.rewards.ang_vel_xy_l2.weight = -0.05
            self.rewards.undesired_contacts.weight = -1.0
            self.rewards.dof_torques_l2.weight = -1.5e-7
            self.rewards.dof_acc_l2.weight = -1.25e-7
            self.rewards.dof_vel_l2.weight = -0.001
            self.rewards.dof_pos_limits.weight = -1.0
            self.rewards.dof_torques_limits.weight = -0.05
            self.rewards.action_rate_l2.weight = -0.05
            self.rewards.flat_orientation_l2.weight = -5.0
            self.rewards.feet_air_time.weight = 0.5
            self.rewards.feet_air_time_variance.weight = -2.0
            self.rewards.feet_slide.weight = -0.3
            self.rewards.feet_stumble.weight = -5.0
            self.rewards.feet_too_near.weight = -5.0
            self.rewards.joint_coordination.weight = -0.5
            self.rewards.joint_deviation_hip.weight = -0.1
            self.rewards.joint_deviation_arms.weight = -0.3
            self.rewards.joint_deviation_waists.weight = -1.0
        else:
            # Randomization
            self.events.push_robot = None
            self.events.add_base_mass = None
            self.events.base_com = None
            # Observations
            self.observations.policy.base_ang_vel.noise=None
            self.observations.policy.projected_gravity.noise=None
            self.observations.policy.velocity_commands.noise=None
            self.observations.policy.joint_pos.noise=None
            self.observations.policy.joint_vel.noise=None
            self.observations.policy.actions.noise=None
            self.observations.policy.height_scan.params["noise"]=False
            # Reward Weights
            self.rewards.termination_penalty.weight = -200
            self.rewards.track_lin_vel_xy_exp.weight = 2.0
            self.rewards.track_ang_vel_z_exp.weight = 3.0
            self.rewards.ang_vel_xy_l2.weight = -0.05
            self.rewards.undesired_contacts.weight = -1.0
            self.rewards.dof_torques_l2.weight = -1.5e-7
            self.rewards.dof_acc_l2.weight = -1.25e-7
            self.rewards.dof_vel_l2.weight = -0.001
            self.rewards.dof_pos_limits.weight = -1.0
            self.rewards.dof_torques_limits.weight = -0.01
            self.rewards.action_rate_l2.weight = -0.01
            self.rewards.flat_orientation_l2.weight = -2.0
            self.rewards.feet_air_time.weight = 0.25
            self.rewards.feet_air_time_variance.weight = -0.7
            self.rewards.feet_slide.weight = -0.1
            self.rewards.feet_stumble.weight = -2.0
            self.rewards.feet_too_near.weight = -1.0
            self.rewards.joint_coordination.weight = -0.2
            self.rewards.joint_deviation_hip.weight = -0.1
            self.rewards.joint_deviation_arms.weight = -0.3
            self.rewards.joint_deviation_waists.weight = -1.0
        

@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        
        # add visualization camera only for play
        self.scene.visualize_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/visualize_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 3.), rot=(0.707, 0.0, 0.707, 0.0), convention="world"),
        )
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        self.events.reset_base.params= {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        }

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.size = (8.0, 8.0)
            self.scene.terrain.terrain_generator.sub_terrains =  {
                # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                #     proportion=0.3,
                #     step_height_range=(0.15, 0.15),
                #     step_width=0.4,
                #     platform_width=3.0,
                #     border_width=1.0,
                #     holes=False,
                # ),
                # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                #     proportion=0.3,
                #     step_height_range=(0.15, 0.15),
                #     step_width=0.3,
                #     platform_width=3.0,
                #     border_width=1.0,
                #     holes=False,
                # ),
                # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                #     proportion=0.2, grid_width=0.45, grid_height_range=(0.1, 0.1), platform_width=2.0
                # ),
                # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                #     proportion=0.1, noise_range=(0.02, 0.1), noise_step=0.01, border_width=0.25
                # ),
                # "hf_steppingstones": terrain_gen.HfSteppingStonesTerrainCfg(
                #     proportion=1.0, stone_height_max=0.0, stone_width_range=(0.3, 0.3), stone_distance_range=(0.2, 0.2), platform_width=2.0,
                #     holes_depth=-2.0, border_width=0.25
                # ),
                # "stonebridge": terrain_gen.HfStonesBridgeTerrainCfg(
                #     proportion=1.0, platform_width=2.0, border_width=0.25, holes_depth=-2.0,
                #     stone_height_max=0.0, stone_width_range=(0.3, 0.3), stone_distance_range=(0.2, 0.2),
                #     stone_length_range=(0.4, 0.4), stone_lateral_distance_range=(0.0, 0.0)
                # ),
                "stakes": terrain_gen.HfAlternateColumnStakesTerrainCfg(
                    proportion=0.5, stake_height_max=0.0, stake_side_range=(0.2, 0.2), stake_gap_range=(0.3, 0.3),
                    column_gap_range=(0.3, 0.3), column_jitter=0.0, holes_depth=-2.0, platform_width=2.0, border_width=0.25
                ),
                # "hf_gaps": terrain_gen.HfConcentricGapTerrainCfg(
                #             proportion=0.5, gap_width_range=(0.5, 0.5), platform_width=2.0, border_width=0.25, gap_depth=-1.0,
                #             ground_width_range=(0.5, 0.5), ground_height_max=0.0
                # ),
                # "rails": terrain_gen.MeshRailsTerrainCfg(
                #     proportion=0.1, rail_height_range=(0.30, 0.30), rail_thickness_range=(0.3, 0.3), platform_width=2.0
                # ),
            }

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"]=False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

