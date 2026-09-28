import gymnasium as gym
for task, cfg in [('Base','Go2BaseEnvCfg'),('Rough','Go2RoughEnvCfg'),('Flat-Play','Go2FlatPlayEnvCfg'),('Obstacle-Play','Go2ObstaclePlayEnvCfg')]:
    gym.register(id=f'AME-Go2-{task}-v0',entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,kwargs={'env_cfg_entry_point':f'{__name__}.env_cfg:{cfg}','rsl_rl_cfg_entry_point':f'{__name__}.runner_cfg:Go2RunnerCfg'})

# Formal AME curriculum: rough foundation, then sparse-support refinement.
for task, cfg in [('Stage1','Go2Stage1EnvCfg'),('Stage2','Go2Stage2EnvCfg'),('Stage1-Play','Go2Stage1PlayEnvCfg'),('Stage2-Play','Go2Stage2PlayEnvCfg')]:
    gym.register(id=f'AME-Go2-{task}-v0',entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,kwargs={'env_cfg_entry_point':f'{__name__}.stages:{cfg}','rsl_rl_cfg_entry_point':f'{__name__}.runner_cfg:Go2RunnerCfg'})
