"""Geometry, RNG, stage coverage and failure/curriculum tests without physics stepping."""
import copy
from types import SimpleNamespace as NS
import numpy as np
import torch
from ame_locomotion.tasks.manager_based.ame_locomotion.go2.complex_terrains import (
    Go2RouteTerrainCfg, stage1_terrain, stage2_terrain, route_height_field, dimensions)
from ame_locomotion.tasks.manager_based.ame_locomotion.go2.stages import (
    column_families, route_curriculum, pit_fall)

def run_checks():
    result={}
    for name, generator in [('stage1',stage1_terrain()),('stage2',stage2_terrain())]:
        assert set(column_families(generator))==set(generator.sub_terrains)
        result[name]={'families':column_families(generator),'routes':{}}
        for family,cfg in generator.sub_terrains.items():
            if not isinstance(cfg,Go2RouteTerrainCfg):continue
            cfg=copy.deepcopy(cfg);cfg.size=(8.,8.);cfg.horizontal_scale=.05;cfg.vertical_scale=.005
            reports=[]
            for d in (0.,.5,1.):
                a=route_height_field(d,cfg)
                b=route_height_field(d,cfg)
                assert np.array_equal(a,b)
                assert a.min()==round(cfg.pit_depth/cfg.vertical_scale)
                assert a[80,80]==0 and np.isfinite(a).all()
                meshes,origin=cfg.function(d,cfg)
                assert all(np.isfinite(m.vertices).all() for m in meshes)
                assert np.isfinite(origin).all()
                reports.append({'difficulty':d,**dimensions(d,cfg),'pit_depth':float(a.min()*cfg.vertical_scale),'pit_fraction':float(np.mean(a==round(cfg.pit_depth/cfg.vertical_scale)))})
            assert reports[0]['gap']<=reports[-1]['gap']
            assert reports[0]['support']>=reports[-1]['support']
            result[name]['routes'][family]=reports
    # Fall into a pit must terminate before landing on its artificial bottom;
    # a normal descending-stair pose must not be mistaken for a pit fall.
    class Scene(dict):pass
    generator=stage2_terrain()
    t=NS(cfg=NS(terrain_generator=generator),terrain_types=torch.tensor([0,2,4,5]))
    robot=NS(data=NS(root_pos_w=torch.tensor([[0.,0.,-.6]]*4)))
    scene=Scene(robot=robot);scene.terrain=t;scene.env_origins=torch.zeros(4,3)
    env=NS(scene=scene,device='cpu')
    assert pit_fall(env).tolist()==[False,True,True,True]
    # Success / early fall / standing / lateral shortcut / cap / no movement.
    t.terrain_levels=torch.tensor([1,1,1,1,5,1]);t.max_terrain_level=6
    t.terrain_types=torch.zeros(6,dtype=torch.long)
    t.terrain_origins=torch.zeros(6,1,3);t.env_origins=torch.zeros(6,3)
    scene.env_origins=t.env_origins
    robot.data.root_pos_w=torch.tensor([[3.6,0.,0.],[3.6,0.,0.],[3.6,0.,0.],[3.6,.9,0.],[3.6,0.,0.],[0.,0.,0.]])
    commands=torch.tensor([[.5,0.,0.]]*6);commands[2]=0
    env.command_manager=NS(get_command=lambda _:commands)
    env.episode_length_buf=torch.full((6,),400);env.step_dt=.02
    env.termination_manager=NS(terminated=torch.tensor([False,True,False,False,False,False]))
    route_curriculum(env,torch.arange(6))
    assert t.terrain_levels.tolist()==[2,0,1,1,5,0]
    result['curriculum_levels']=t.terrain_levels.tolist()
    return result
