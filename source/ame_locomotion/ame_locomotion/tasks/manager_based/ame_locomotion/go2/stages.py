"""Formal AME stages. Legacy flat tasks remain available for diagnostics."""
import torch
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm, CurriculumTermCfg as CurrTerm
from .env_cfg import Go2RoughEnvCfg, evaluation
from .complex_terrains import stage1_terrain, stage2_terrain


def column_families(generator):
    import numpy as np
    names = list(generator.sub_terrains)
    proportions = np.array([x.proportion for x in generator.sub_terrains.values()])
    cumulative = np.cumsum(proportions / proportions.sum())
    return [names[int(np.where(col / generator.num_cols + .001 < cumulative)[0][0])]
            for col in range(generator.num_cols)]


def pit_fall(env):
    """Do not mistake descending stairs/slopes for falling into a negative obstacle."""
    terrain = env.scene.terrain
    families = column_families(terrain.cfg.terrain_generator)
    pit_columns = torch.tensor([name in ('double_stakes','alternate_stakes','bridge','gaps','stepping_stones')
                                for name in families], device=env.device)
    height = env.scene['robot'].data.root_pos_w[:,2] - env.scene.env_origins[:,2]
    return pit_columns[terrain.terrain_types] & (height < -0.45)


def tile_complete(env):
    displacement = env.scene['robot'].data.root_pos_w - env.scene.env_origins
    return displacement[:,0] > 3.5


def route_departure(env):
    displacement = env.scene['robot'].data.root_pos_w - env.scene.env_origins
    return (displacement[:,1].abs() > 1.) | (displacement[:,0] < -3.5)


def route_curriculum(env, env_ids):
    t = env.scene.terrain
    delta = env.scene['robot'].data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    elapsed = env.episode_length_buf[env_ids] * env.step_dt
    command = env.command_manager.get_command('base_velocity')[env_ids,0]
    moving = command > .1
    alive = ~env.termination_manager.terminated[env_ids]
    # Require genuine route completion/progress, no radial-distance shortcut.
    up = moving & alive & (elapsed > 2.) & (delta[:,0] > 3.) & (delta[:,1].abs() < .8) & (delta[:,0] > .65*command*elapsed)
    down = moving & (elapsed > 1.) & (~alive | (delta[:,0] < .3*command*elapsed))
    t.terrain_levels[env_ids] = (t.terrain_levels[env_ids] + up.long() - (down & ~up).long()).clamp(0,t.max_terrain_level-1)
    t.env_origins[env_ids] = t.terrain_origins[t.terrain_levels[env_ids],t.terrain_types[env_ids]]
    return t.terrain_levels.float().mean()


@configclass
class Go2Stage1EnvCfg(Go2RoughEnvCfg):
    ame_stage: str = 'rough_foundation_v1'
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = stage1_terrain()
        self.commands.base_velocity.rel_standing_envs = .05
        self.observations.policy.enable_corruption = False
        self.events.physics_material.params.update(static_friction_range=(.8,.8),dynamic_friction_range=(.6,.6))
        self.curriculum.terrain_levels = CurrTerm(func=route_curriculum)
        self.terminations.pit_fall = DoneTerm(func=pit_fall)
        self.terminations.tile_complete = DoneTerm(func=tile_complete,time_out=True)
        self.terminations.route_departure = DoneTerm(func=route_departure)


@configclass
class Go2Stage2EnvCfg(Go2Stage1EnvCfg):
    ame_stage: str = 'complex_refinement_v1'
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = stage2_terrain()
        self.observations.policy.enable_corruption = True
        self.events.physics_material.params.update(static_friction_range=(.6,1.),dynamic_friction_range=(.5,.6))


@configclass
class Go2Stage1PlayEnvCfg(Go2Stage1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        evaluation(self)


@configclass
class Go2Stage2PlayEnvCfg(Go2Stage2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        evaluation(self)
