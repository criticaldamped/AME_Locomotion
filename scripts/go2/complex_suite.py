"""Bounded integration checks for the formal rough -> complex AME stages."""
import argparse
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
stage1=root/'stage1/model_3.pt'
stage2=root/'stage2/model_3.pt'
runs=[
    ('stage1', ['--mode','train','--task','AME-Go2-Stage1-v0','--num_envs','64','--iterations','4']),
    ('stage2', ['--mode','train','--task','AME-Go2-Stage2-v0','--num_envs','64','--iterations','4','--warm_start','--checkpoint',str(stage1)]),
    ('resume2', ['--mode','train','--task','AME-Go2-Stage2-v0','--num_envs','64','--iterations','2','--resume','--checkpoint',str(stage2)]),
]
for family in ('double_stakes','alternate_stakes','bridge','gaps'):
    runs.append((family,['--mode','play','--task','AME-Go2-Stage2-Play-v0','--num_envs','8','--steps','250','--level','3','--terrain',family,'--velocity','.5','--checkpoint',str(stage2)]))
runs.append(('g1',['--mode','probe','--task','AME-G1-29DOF-v0','--num_envs','2','--steps','20']))
for name,args in runs:
    with (root/f'{name}.log').open('w') as log:
        result=subprocess.run([sys.executable,'-u','scripts/go2/run.py','--headless','--output',str(root/name),*args],stdout=log,stderr=subprocess.STDOUT,timeout=300)
    success=result.returncode==0 and (root/name/'SUCCESS').exists()
    print(name, 'PASS' if success else 'FAIL',flush=True)
    if not success:sys.exit(1)
# Both stages are generators, but their curriculum state must not be confused.
with (root/'reject_cross_stage.log').open('w') as log:
    result=subprocess.run([sys.executable,'-u','scripts/go2/run.py','--headless','--mode','train','--task','AME-Go2-Stage2-v0','--num_envs','64','--iterations','1','--resume','--checkpoint',str(stage1),'--output',str(root/'reject_cross_stage')],stdout=log,stderr=subprocess.STDOUT,timeout=300)
assert result.returncode==1
assert 'Stage/configuration changed' in (root/'reject_cross_stage/FAILED').read_text()
print('cross-stage resume rejected: PASS',flush=True)
