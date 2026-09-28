"""Re-run a bounded integration suite; never starts a long training."""
import argparse,subprocess,sys,time,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args()
root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
runs=[
 ('resume64',['--mode','train','--task','AME-Go2-Base-v0','--num_envs','64','--iterations','2','--checkpoint',a.checkpoint,'--resume']),
 ('flat_eval',['--mode','play','--task','AME-Go2-Flat-Play-v0','--num_envs','16','--steps','500','--checkpoint',a.checkpoint,'--export']),
 ('obstacle0',['--mode','play','--task','AME-Go2-Obstacle-Play-v0','--num_envs','16','--steps','500','--level','0','--checkpoint',a.checkpoint]),
 ('obstacle2',['--mode','play','--task','AME-Go2-Obstacle-Play-v0','--num_envs','16','--steps','500','--level','2','--checkpoint',a.checkpoint]),
 ('rough_train',['--mode','train','--task','AME-Go2-Rough-v0','--num_envs','64','--iterations','2','--checkpoint',a.checkpoint,'--warm_start']),
 ('g1_regression',['--mode','probe','--task','AME-G1-29DOF-v0','--num_envs','2','--steps','20']),
 ('throughput512',['--mode','train','--task','AME-Go2-Base-v0','--num_envs','512','--iterations','4']),
]
for name,args in runs:
    out=root/name
    if out.exists():raise FileExistsError(out)
    with (root/(name+'.log')).open('w') as log,(root/(name+'_gpu.csv')).open('w') as gpu:
        proc=subprocess.Popen([sys.executable,'-u','scripts/go2/run.py','--headless','--output',str(out),*args],stdout=log,stderr=subprocess.STDOUT)
        start=time.monotonic()
        while proc.poll() is None:
            gpu.write(str(time.time())+','+subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True));gpu.flush()
            if time.monotonic()-start>300:proc.terminate();time.sleep(2);proc.kill();break
            time.sleep(2)
    print(name,proc.returncode,'PASS' if (out/'SUCCESS').exists() else 'FAIL',flush=True)
