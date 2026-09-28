"""Go2 integration with the project's RSL-RL runner; versioned checkpoint contract."""
import copy
import hashlib
import json
import random
import numpy as np
import torch
from pathlib import Path
from rsl_rl.modules.go2_actor_critic import Go2ActorCritic, finite
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.runners import on_policy_runner as runner_module

class CheckedPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finite_checks=True
        self.min_learning_rate=1e-5
        self.max_learning_rate=3e-4
        self.rollout_stats=[]
        self.last_rollout={}
    def act(self, obs):
        actions=super().act(obs)
        if hasattr(self, "environment"):
            e=self.environment
            a=e.action_manager.get_term("joint_pos")
            robot=e.scene["robot"]
            self.rollout_stats.append(torch.stack((actions.abs().max(), actions.square().mean(), ((actions<a.lower)|(actions>a.upper)).float().mean(), robot.data.root_lin_vel_b[:,0].mean(), e.command_manager.get_command("base_velocity")[:,0].mean())))
        return actions
    def update(self):
        result=super().update()
        if self.rollout_stats:
            stats=torch.stack(self.rollout_stats)
            self.last_rollout={"raw_action_abs_max":stats[:,0].max().item(),"raw_action_rms":stats[:,1].mean().sqrt().item(),"clip_fraction":stats[:,2].mean().item(),"vx":stats[:,3].mean().item(),"command_vx":stats[:,4].mean().item(),"last_gradient_norm":self.last_grad_norm.item()}
            self.rollout_stats.clear()
        std=self.policy.log_std.exp()
        if not ((std>=0.02)&(std<=2.0)).all():
            raise FloatingPointError("Policy std outside [0.02,2.0] after update")
        return result
    def process_env_step(self, obs, rewards, dones, extras):
        for key, value in obs.items():
            finite(f'observation/{key}', value)
        finite('reward', rewards)
        return super().process_env_step(obs, rewards, dones, extras)
    def compute_returns(self, obs):
        super().compute_returns(obs)
        finite('returns', self.storage.returns)
        finite('advantages', self.storage.advantages)

# The upstream runner resolves class names in this namespace. G1 classes are untouched.
runner_module.Go2ActorCritic = Go2ActorCritic
runner_module.CheckedPPO = CheckedPPO

class Go2Runner(runner_module.OnPolicyRunner):
    def __init__(self, env, cfg, **kwargs):
        self.saved_cfg=copy.deepcopy(cfg)
        super().__init__(env, cfg, **kwargs)
        self.alg.environment=self.env.unwrapped
        print("[Go2] Final CNN normalization:", [type(m).__name__ for m in self.alg.policy.map_cnn if isinstance(m, torch.nn.GroupNorm)],flush=True)
        self.logger_type=cfg.get('logger','tensorboard')
        self.metrics_path=Path(self.log_dir)/'iterations.jsonl' if self.log_dir else None
    def contract(self):
        e=self.env.unwrapped
        a=e.action_manager.get_term('joint_pos')
        return {'version':2, 'source_sha256':{str(p.relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve().parents[2]/'rsl_rl/rsl_rl/modules/go2_actor_critic.py',Path(__file__).resolve().parents[2]/'source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/go2/mdp.py',Path(__file__).resolve().parents[2]/'source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/go2/env_cfg.py']}, 'dt':e.step_dt,'default_pose':a._offset[0].tolist(), 'kp':e.scene['robot'].data.joint_stiffness[0].tolist(), 'kd':e.scene['robot'].data.joint_damping[0].tolist(),'joints':a._joint_names,'scale':a._scale,'lower':a.lower[0].tolist(),'upper':a.upper[0].tolist(),'map':[33,21,3],'map_order':'xy/y-first rows, x-fast','origin':'base yaw','obs':[2124,2127],'normalization':'GroupNorm4-fixed-proprio-scales','policy':self.saved_cfg['policy']}
    def stage_contract(self):
        e = self.env.unwrapped
        if not hasattr(e.cfg, 'ame_stage'):
            return None
        def encode(value):
            if callable(value):
                return f'{getattr(value, "__module__", type(value).__module__)}.{getattr(value, "__qualname__", type(value).__qualname__)}'
            return str(value)
        configuration = {name:getattr(e.cfg, name).to_dict() for name in
                         ('commands','rewards','events','terminations','curriculum')}
        configuration['terrain'] = e.cfg.scene.terrain.terrain_generator.to_dict()
        folder = Path(__file__).resolve().parents[2] / 'source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/go2'
        return {'stage':e.cfg.ame_stage,
                'configuration_sha256':hashlib.sha256(json.dumps(configuration,sort_keys=True,default=encode).encode()).hexdigest(),
                'source_sha256':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ('stages.py','complex_terrains.py')}}
    def save(self, path, infos=None):
        e=self.env.unwrapped
        data={'model_state_dict':self.alg.policy.state_dict(),'optimizer_state_dict':self.alg.optimizer.state_dict(),'iter':self.current_learning_iteration+1,'learning_rate':self.alg.learning_rate,'contract':self.contract(),'train_cfg':self.saved_cfg,'infos':infos,'stage_contract':self.stage_contract(),'task_stage':e.cfg.scene.terrain.terrain_type,'seed':e.cfg.seed,'num_envs':e.num_envs,'common_step_counter':e.common_step_counter,'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()},'terrain_levels':e.scene.terrain.terrain_levels.cpu() if e.cfg.scene.terrain.terrain_type=='generator' else None}
        # Atomic replacement inside a newly-created run directory only.
        temp=str(path)+'.tmp'
        torch.save(data,temp)
        Path(temp).replace(path)
        print(f"[Go2] Checkpoint saved: {Path(path).resolve()}",flush=True)
    def load(self,path,load_optimizer=True,map_location=None):
        d=torch.load(path,weights_only=False,map_location=map_location or self.device)
        if d.get('contract') != self.contract():
            raise ValueError('Checkpoint contract mismatch (Go2 model, joint/map order or normalization)')
        self.alg.policy.load_state_dict(d['model_state_dict'])
        if load_optimizer:
            e=self.env.unwrapped
            if d.get('stage_contract') != self.stage_contract():
                raise ValueError('Stage/configuration changed: use --warm_start, not --resume')
            if d['task_stage'] != e.cfg.scene.terrain.terrain_type:
                raise ValueError('Stage change requires --warm_start, not --resume')
            if d['seed'] != e.cfg.seed:
                raise ValueError('Resume requires the saved seed; use warm_start for a new seed')
            if d['num_envs'] != e.num_envs:
                raise ValueError('Resume requires same num_envs; use warm_start for changed batch size')
            self.alg.optimizer.load_state_dict(d['optimizer_state_dict'])
            self.alg.learning_rate=float(d['learning_rate'])
            if not 1e-5 <= self.alg.learning_rate <= 3e-4:
                raise ValueError('Checkpoint learning rate out of bounds')
            self.current_learning_iteration=d['iter']
            e.common_step_counter=d['common_step_counter']
            if d['terrain_levels'] is not None:
                levels=d['terrain_levels'].to(e.device)
                if levels.shape != e.scene.terrain.terrain_levels.shape:
                    raise ValueError('Resume terrain state requires same num_envs; use warm_start otherwise')
                t=e.scene.terrain
                t.terrain_levels[:]=levels
                t.env_origins[:]=t.terrain_origins[levels,t.terrain_types]
                e.scene.env_origins[:]=t.env_origins
                e.reset()
            torch.set_rng_state(d['rng']['torch'].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in d['rng']['cuda']])
            np.random.set_state(d['rng']['numpy']);random.setstate(d['rng']['python'])
        if self.log_dir:
            Path(self.log_dir,'loaded_checkpoint.json').write_text(json.dumps({'path':str(Path(path).resolve()),'load_optimizer':load_optimizer,'next_iteration':self.current_learning_iteration,'learning_rate':self.alg.learning_rate,'terrain_levels_restored':load_optimizer and d['terrain_levels'] is not None},indent=2))
        return d.get('infos')
    def log(self,locs,*args,**kwargs):
        super().log(locs,*args,**kwargs)
        e=self.env.unwrapped
        a=e.action_manager.get_term('joint_pos')
        metrics={'iteration':locs['it'],'steps_per_s':self.num_steps_per_env*self.env.num_envs/(locs['collection_time']+locs['learn_time']),'lr':self.alg.learning_rate,'std_min':self.alg.policy.action_std.min().item(),'std_max':self.alg.policy.action_std.max().item(),'raw_action_abs_max_last_step':a.raw_actions.abs().max().item(),'clip_fraction_last_step':a.clip_fraction.mean().item(),'loss':locs['loss_dict'],'rollout':self.alg.last_rollout}
        for name,value in self.alg.last_rollout.items():self.writer.add_scalar('Diagnostics/'+name,value,locs['it'])
        with self.metrics_path.open('a') as f:f.write(json.dumps(metrics)+'\n')
