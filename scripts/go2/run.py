"""Bounded diagnostics, training and deterministic evaluation for Go2 AME."""
import _bootstrap
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['train','play','probe','config'],default='play')
p.add_argument('--task',default='AME-Go2-Flat-Play-v0')
p.add_argument('--num_envs',type=int,default=16)
p.add_argument('--iterations',type=int,default=12,help='Additional iterations, also on resume')
p.add_argument('--steps',type=int,default=500)
p.add_argument('--checkpoint')
g=p.add_mutually_exclusive_group();g.add_argument('--resume',action='store_true');g.add_argument('--warm_start',action='store_true')
p.add_argument('--seed',type=int,default=42)
p.add_argument('--terrain',default=None,help='Single terrain family for play/probe, e.g. bridge, gaps, double_stakes')
p.add_argument('--level',type=int,default=0)
p.add_argument('--velocity',type=float,default=0.5)
p.add_argument('--output',default=None)
p.add_argument('--export',action='store_true')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.iterations < 1 or a.steps < 1 or a.num_envs < 1: p.error('Counts must be positive')
if (a.resume or a.warm_start) and not a.checkpoint:p.error('Explicit checkpoint required')
if a.mode=='play' and not a.checkpoint:p.error('Playback requires --checkpoint')
app=AppLauncher(a).app
import torch
import gymnasium as gym
import ame_locomotion.tasks
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml
from runner import Go2Runner
from rsl_rl.modules.go2_actor_critic import finite

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=True
output_root = _bootstrap.ROOT / ('logs/rsl_rl/go2_ame' if a.mode == 'train' else 'diagnostics')
out=Path(a.output) if a.output else output_root/('go2_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
out.mkdir(parents=True,exist_ok=False)
print(f'[Go2] Output directory: {out.resolve()}',flush=True)
(out/'arguments.json').write_text(json.dumps(vars(a),indent=2))

def main():
    cfg=load_cfg_from_registry(a.task,'env_cfg_entry_point')
    agent=load_cfg_from_registry(a.task,'rsl_rl_cfg_entry_point')
    cfg.scene.num_envs=a.num_envs
    cfg.seed=a.seed
    cfg.sim.device=a.device or 'cuda:0'
    agent.device=cfg.sim.device
    agent.seed=a.seed
    if a.terrain:
        if a.mode=='train':raise ValueError('--terrain is for evaluation/probe only')
        generator=cfg.scene.terrain.terrain_generator
        if generator is None or a.terrain not in generator.sub_terrains:raise ValueError('Unknown terrain family')
        generator.sub_terrains={a.terrain:generator.sub_terrains[a.terrain]}
        generator.sub_terrains[a.terrain].proportion=1.
        generator.num_cols=1
    if a.mode=='config':
        from pxr import Usd,UsdPhysics
        from mdp_checks import run_checks
        rows={'reward_curriculum_checks':run_checks()}
        for task in ['AME-G1-29DOF-v0','AME-G1-29DOF-Play-v0']+[name for name in gym.registry if name.startswith('AME-Go2-')]:
            c=load_cfg_from_registry(task,'env_cfg_entry_point');c.validate()
            ag=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point')
            if 'Go2' in task: ag.validate()
            rows[task]={'asset':c.scene.robot.spawn.usd_path,'asset_exists':Path(c.scene.robot.spawn.usd_path).is_file()}
        from complex_checks import run_checks as complex_checks
        rows['complex_terrain_checks']=complex_checks()
        stage=Usd.Stage.Open(cfg.scene.robot.spawn.usd_path)
        rows['usd_dependencies']=[x.realPath for x in stage.GetUsedLayers()]
        rows['usd_joints']={str(prim.GetName()):{'lower_degrees':UsdPhysics.RevoluteJoint(prim).GetLowerLimitAttr().Get(),'upper_degrees':UsdPhysics.RevoluteJoint(prim).GetUpperLimitAttr().Get()} for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)}
        (out/'config.json').write_text(json.dumps(rows,indent=2));print(rows);return
    if 'Go2' in a.task and a.mode!='train':
        cfg.commands.base_velocity.rel_standing_envs=0.
        cfg.commands.base_velocity.heading_command=False
        cfg.commands.base_velocity.ranges.lin_vel_x=(a.velocity,a.velocity)
        cfg.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
        cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
        cfg.observations.policy.enable_corruption=False
    dump_yaml(str(out/'env.yaml'),cfg)
    dump_yaml(str(out/'agent.yaml'),agent)
    if hasattr(cfg, 'ame_stage'):
        from ame_locomotion.tasks.manager_based.ame_locomotion.go2.stages import column_families
        (out/'terrain_manifest.json').write_text(json.dumps({'stage':cfg.ame_stage,'columns':column_families(cfg.scene.terrain.terrain_generator),'num_rows':cfg.scene.terrain.terrain_generator.num_rows,'seed':cfg.scene.terrain.terrain_generator.seed},indent=2))
    env=gym.make(a.task,cfg=cfg)
    e=env.unwrapped
    if cfg.scene.terrain.terrain_type=='generator' and a.mode!='train':
        t=e.scene.terrain
        if not 0<=a.level<t.cfg.terrain_generator.num_rows:raise ValueError('Terrain level out of range')
        t.terrain_levels[:]=a.level
        t.env_origins[:]=t.terrain_origins[t.terrain_levels,t.terrain_types]
        e.scene.env_origins[:]=t.env_origins
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    obs=wrapped.get_observations()
    robot=e.scene['robot']
    action=e.action_manager.get_term('joint_pos')
    if 'Go2' not in a.task:
        for _ in range(min(a.steps,50)):
            obs,reward,done,info=wrapped.step(torch.zeros((a.num_envs,wrapped.num_actions),device=e.device))
            for key,v in obs.items():finite(key,v)
        (out/'g1_regression.json').write_text(json.dumps({'steps':min(a.steps,50),'shapes':{k:list(v.shape) for k,v in obs.items()},'finite':True}));env.close();return
    assert obs['policy'].shape==(a.num_envs,2124)
    assert obs['critic'].shape==(a.num_envs,2127)
    names=robot.joint_names
    metadata={'joint_order':action._joint_names,'asset_joint_order':names,'body_names':robot.body_names,'default_joint_pos':action._offset[0].tolist(),'target_limits':action.limits[0].tolist(),'raw_action_lower':action.lower[0].tolist(),'raw_action_upper':action.upper[0].tolist(),'kp':robot.data.joint_stiffness[0].tolist(),'kd':robot.data.joint_damping[0].tolist(),'effort_limits':robot.data.joint_effort_limits[0].tolist(),'velocity_limits':robot.data.joint_vel_limits[0].tolist(),'shapes':{k:list(v.shape) for k,v in obs.items()}}
    metadata['physx_kp']=metadata.pop('kp')
    metadata['physx_kd']=metadata.pop('kd')
    metadata['explicit_actuators']={name:{'kp':motor.stiffness[0].tolist(),'kd':motor.damping[0].tolist(),'effort_limit':motor.effort_limit[0].tolist(),'velocity_limit':motor.velocity_limit[0].tolist()} for name,motor in robot.actuators.items()}
    assert all(torch.all(m.stiffness==25.) and torch.all(m.damping==0.5) for m in robot.actuators.values())
    (out/'robot_contract.json').write_text(json.dumps(metadata,indent=2))
    if a.mode=='probe':
        for magnitude in [-1000.,1000.]:
            action.process_actions(torch.full_like(action.raw_actions,magnitude))
            assert ((action.processed_actions>=action.limits[...,0])&(action.processed_actions<=action.limits[...,1])).all()
            assert (action.clip_fraction==1).all()
        action.process_actions(torch.zeros_like(action.raw_actions))
    runner=Go2Runner(wrapped,agent.to_dict(),log_dir=str(out),device=e.device)
    if a.checkpoint:runner.load(a.checkpoint,load_optimizer=a.resume)
    if a.mode=='train':
        if a.checkpoint and not (a.resume or a.warm_start):raise ValueError('Choose --resume or --warm_start explicitly')
        start=time.monotonic()
        runner.learn(a.iterations,init_at_random_ep_len=False)
        (out/'train_summary.json').write_text(json.dumps({'elapsed_s':time.monotonic()-start,'additional_iterations':a.iterations,'cuda_peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'cuda_peak_reserved_gb':torch.cuda.max_memory_reserved()/1e9},indent=2))
        env.close();return
    if a.export:
        from export import export_policy
        export_policy(runner.alg.policy,obs['policy'],out)
    policy=runner.get_inference_policy()
    obs=wrapped.get_observations()
    resets=falls=0
    displacement=torch.zeros(a.num_envs,device=e.device)
    rewards=torch.zeros(len(e.reward_manager.active_terms),device=e.device)
    totals={'command':torch.zeros(3,device=e.device),'velocity':torch.zeros(3,device=e.device),'clip':0.,'raw_abs_max':0.}
    force_max=0.;height_min=100.;height_max=-100.
    start=time.monotonic()
    with torch.inference_mode(), (out/'steps.jsonl').open('w') as stream:
        for step in range(a.steps):
            if a.mode=='probe':
                actions=torch.zeros((a.num_envs,12),device=e.device)
                if step>=100:actions[:,1::3]=0.1*torch.sin(torch.tensor((step-100)*0.08,device=e.device))
            else:actions=policy(obs)
            pos=robot.data.root_pos_w[:,0].clone()
            vel=torch.cat((robot.data.root_lin_vel_b[:,:2],robot.data.root_ang_vel_b[:,2:3]),dim=-1).mean(0).clone()
            cmd=e.command_manager.get_command('base_velocity').mean(0).clone()
            obs,rew,done,info=wrapped.step(actions)
            for key,v in obs.items():finite(key,v)
            finite('joint position',robot.data.joint_pos);finite('torque',robot.data.applied_torque)
            finite('reward',rew)
            assert robot.data.applied_torque.abs().max()<=23.5001
            displacement+=torch.where(done.bool(),0.,robot.data.root_pos_w[:,0]-pos)
            resets+=int(done.sum());falls+=int(e.termination_manager.terminated.sum())
            rewards+=e.reward_manager._step_reward.mean(0)
            clip=((actions<action.lower)|(actions>action.upper)).float().mean().item()
            totals['command']+=cmd;totals['velocity']+=vel;totals['clip']+=clip;totals['raw_abs_max']=max(totals['raw_abs_max'],actions.abs().max().item())
            h=obs['policy'][:,-2079:].reshape(a.num_envs,21,33,3)
            # Verify x-fast/y-row spatial contract, at every step.
            assert torch.allclose(h[:,:,1:,0]-h[:,:,:-1,0],torch.full_like(h[:,:,1:,0],0.05),atol=2e-4)
            height_min=min(height_min,h[...,2].min().item());height_max=max(height_max,h[...,2].max().item())
            assert torch.allclose(h[:,1:,:,1]-h[:,:-1,:,1],torch.full_like(h[:,1:,:,1],0.05),atol=2e-4)
            torch.testing.assert_close(obs['policy'][:,33:45],e.action_manager.action,rtol=0,atol=0)
            assert ((action.processed_actions >= action.limits[...,0]) & (action.processed_actions <= action.limits[...,1])).all()
            force_max=max(force_max,e.scene['contact_forces'].data.net_forces_w.norm(dim=-1).max().item())
            if step%25==0:
                row={'step':step,'cmd':cmd.tolist(),'velocity':vel.tolist(),'raw_abs_max':actions.abs().max().item(),'clip_fraction':clip,'resets':resets,'falls':falls,'forward_displacement_mean':displacement.mean().item(),'reward_terms':dict(zip(e.reward_manager.active_terms,e.reward_manager._step_reward.mean(0).tolist()))}
                stream.write(json.dumps(row)+'\n');stream.flush()
    result={'task':a.task,'terrain_family':a.terrain,'checkpoint':a.checkpoint,'mode':a.mode,'num_envs':a.num_envs,'steps':a.steps,'sim_seconds_per_env':a.steps*e.step_dt,'elapsed_s':time.monotonic()-start,'command_mean':(totals['command']/a.steps).tolist(),'velocity_xy_yaw_mean':(totals['velocity']/a.steps).tolist(),'forward_displacement_excluding_reset_steps_per_env':displacement.tolist(),'falls':falls,'resets':resets,'raw_action_abs_max':totals['raw_abs_max'],'action_clip_fraction':totals['clip']/a.steps,'std':runner.alg.policy.log_std.exp().tolist(),'reward_rate_mean':dict(zip(e.reward_manager.active_terms,(rewards/a.steps).tolist())),'terrain_level':a.level if cfg.scene.terrain.terrain_type=='generator' else None,'height_z_range':[height_min,height_max],'final_map_height_span':(h[...,2].amax((1,2))-h[...,2].amin((1,2))).tolist(),'final_base_height':(robot.data.root_pos_w[:,2]-e.scene.env_origins[:,2]).tolist(),'final_joint_pos':robot.data.joint_pos[0].tolist(),'contact_force_max':force_max,'nonfinite_count':0,'cuda_peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    env.close()
try:
    main()
except BaseException:
    import traceback,sys
    traceback.print_exc()
    sys.stdout.flush();sys.stderr.flush()
    (out/"FAILED").write_text(traceback.format_exc())
    import os
    os._exit(1)  # Isaac Sim fast shutdown otherwise masks the failure exit status.
else:
    import sys
    sys.stdout.flush()
    (out/"SUCCESS").write_text("completed\n")
    app.close()
