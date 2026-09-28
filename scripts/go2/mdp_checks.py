"""Tensor-level reward/curriculum checks, after the simulator supplies Isaac Lab imports."""
from types import SimpleNamespace as NS
import torch
from ame_locomotion.tasks.manager_based.ame_locomotion.go2 import mdp

def run_checks():
    robot=NS(data=NS(root_lin_vel_b=torch.tensor([[0.,0.,0.],[.5,0.,0.],[0.,0.,0.]]),root_ang_vel_b=torch.zeros(3,3)))
    commands=torch.tensor([[.5,0.,0.],[.5,0.,0.],[0.,0.,0.]])
    env=NS(scene={'robot':robot},command_manager=NS(get_command=lambda _:commands))
    lin=mdp.track_linear(env,.35);yaw=mdp.track_yaw(env,.5)
    assert lin[0]==0 and lin[1]>.8 and lin[2]==1
    assert yaw.tolist()==[0.,1.,1.]
    class Scene(dict):pass
    t=NS(terrain_levels=torch.tensor([1,1,1,5,1,1]),max_terrain_level=6,terrain_types=torch.zeros(6,dtype=torch.long),terrain_origins=torch.zeros(6,1,3),env_origins=torch.zeros(6,3))
    robot.data.root_pos_w=torch.tensor([[4.,0.,0.],[6.,0.,0.],[0.,0.,0.],[4.,0.,0.],[0.,4.,0.],[0.,0.,0.]])
    scene=Scene(robot=robot);scene.terrain=t;scene.env_origins=t.env_origins
    commands=torch.tensor([[.5,0.,0.]]*6);commands[2]=0
    env=NS(scene=scene,command_manager=NS(get_command=lambda _:commands),episode_length_buf=torch.tensor([500,500,500,500,500,0]),step_dt=.02,termination_manager=NS(terminated=torch.tensor([False,True,False,False,False,False])))
    mdp.terrain_progress(env,torch.arange(6))
    assert t.terrain_levels.tolist()==[2,0,1,5,0,1],t.terrain_levels
    return {'stationary_moving_command_linear_reward':lin[0].item(),'stationary_moving_command_yaw_reward':yaw[0].item(),'curriculum_expected_levels':t.terrain_levels.tolist()}
