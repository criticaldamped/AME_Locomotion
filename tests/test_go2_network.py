"""CPU-only contract tests, no simulator required. Run with the pinned AME environment."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'rsl_rl'),str(ROOT/'scripts/go2')]
import unittest
import torch
from tensordict import TensorDict
from rsl_rl.modules.go2_actor_critic import Go2ActorCritic
from rsl_rl.modules.actor_critic_encoder import ActorCriticEncoder
from runner import CheckedPPO
from export import export_policy
import tempfile

torch.set_num_threads(2)
def observations(n=8):
    return TensorDict({'policy':torch.randn(n,2124),'critic':torch.randn(n,2127)},[n])
def policy(obs):
    return Go2ActorCritic(obs,{'policy':['policy'],'critic':['critic']},12,noise_std_type='log',init_noise_std=.5,actor_hidden_dims=[32],critic_hidden_dims=[32])
class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.obs=observations()
        self.p=policy(self.obs)
    def test_shapes_modes_and_batch_independence(self):
        self.assertEqual(self.p.act(self.obs).shape,(8,12))
        self.assertEqual(self.p.evaluate(self.obs).shape,(8,1))
        train=self.p.act_inference(self.obs)
        self.p.eval()
        torch.testing.assert_close(train,self.p.act_inference(self.obs),rtol=0,atol=0)
        torch.testing.assert_close(train[:1],self.p.act_inference(self.obs[:1]),atol=1e-6,rtol=1e-5)
    def test_nonfinite_and_noise_fail_fast(self):
        self.obs['policy'][0,50]=float('nan')
        with self.assertRaises(FloatingPointError):self.p.act(self.obs)
        self.obs=observations()
        with torch.no_grad():self.p.log_std.fill_(2)
        with self.assertRaises(FloatingPointError):self.p.act(self.obs)
    def test_ppo_roundtrip_and_gradient_guard(self):
        alg=CheckedPPO(self.p,num_learning_epochs=1,num_mini_batches=2,learning_rate=3e-4,schedule='adaptive',device='cpu')
        alg.init_storage('rl',8,2,self.obs,[12])
        with torch.no_grad():
            for _ in range(2):
                actions=alg.act(self.obs)
                # Log probability is on raw Gaussian sample, before environment target projection.
                torch.testing.assert_close(self.p.get_actions_log_prob(actions),self.p.distribution.log_prob(actions).sum(-1))
                alg.process_env_step(self.obs,torch.ones(8),torch.zeros(8),{})
            alg.compute_returns(self.obs)
        losses=alg.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        self.assertLessEqual(alg.learning_rate,3e-4)
        # Inject a non-finite loss through returns; optimizer must not change parameters.
        with torch.no_grad():
            for _ in range(2):
                alg.act(self.obs);alg.process_env_step(self.obs,torch.ones(8),torch.zeros(8),{})
            alg.compute_returns(self.obs)
        alg.storage.returns.fill_(float('nan'))
        before={k:v.clone() for k,v in self.p.state_dict().items()}
        with self.assertRaises(FloatingPointError):alg.update()
        for k,v in self.p.state_dict().items():torch.testing.assert_close(v,before[k])
    def test_gradient_nonfinite_stops_before_optimizer(self):
        alg=CheckedPPO(self.p,num_learning_epochs=1,num_mini_batches=1,device='cpu')
        alg.init_storage('rl',8,2,self.obs,[12])
        with torch.no_grad():
            for _ in range(2):
                alg.act(self.obs);alg.process_env_step(self.obs,torch.ones(8),torch.zeros(8),{})
            alg.compute_returns(self.obs)
        hook=next(self.p.parameters()).register_hook(lambda grad:grad*float('inf'))
        before={k:v.clone() for k,v in self.p.state_dict().items()}
        with self.assertRaises(RuntimeError):alg.update()
        hook.remove()
        for k,v in self.p.state_dict().items():torch.testing.assert_close(v,before[k])
    def test_g1_ppo_matches_prechange(self):
        import importlib.util,copy
        from rsl_rl.algorithms.ppo import PPO
        path=ROOT/'diagnostics/go2_20260927/original/rsl_rl/rsl_rl/algorithms/ppo.py'
        spec=importlib.util.spec_from_file_location('original_ppo',path)
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        gobs=TensorDict({'policy':torch.randn(8,2175),'critic':torch.randn(8,2178)},[8])
        gp=ActorCriticEncoder(gobs,{'policy':['policy'],'critic':['critic']},29,actor_hidden_dims=[32],critic_hidden_dims=[32])
        algorithms=[cls(copy.deepcopy(gp),num_learning_epochs=1,num_mini_batches=2,device='cpu') for cls in [original.PPO,PPO]]
        losses=[]
        for alg in algorithms:
            torch.manual_seed(77)
            alg.init_storage('rl',8,2,gobs,[29])
            with torch.no_grad():
                for _ in range(2):
                    alg.act(gobs);alg.process_env_step(gobs,torch.ones(8),torch.zeros(8),{})
                alg.compute_returns(gobs)
            losses.append(alg.update())
        self.assertEqual(losses[0],losses[1])
        for k,v in algorithms[0].policy.state_dict().items():
            torch.testing.assert_close(v,algorithms[1].policy.state_dict()[k],atol=0,rtol=0)
    def test_supplied_g1_checkpoints(self):
        gobs=TensorDict({'policy':torch.randn(2,2175),'critic':torch.randn(2,2178)},[2])
        for filename,attach_global in [('ame1.pt',False),('ame2.pt',True)]:
            gp=ActorCriticEncoder(gobs,{'policy':['policy'],'critic':['critic']},29,attach_global=attach_global)
            gp.load_state_dict(torch.load(ROOT/'pretrained'/filename,map_location='cpu',weights_only=False)['model_state_dict'])
            gp.eval()
            self.assertTrue(torch.isfinite(gp.act_inference(gobs)[0]).all())
    def test_export(self):
        with tempfile.TemporaryDirectory() as tmp:export_policy(self.p,self.obs['policy'],Path(tmp))
    def test_g1_shape_state_compatibility(self):
        gobs=TensorDict({'policy':torch.randn(2,2175),'critic':torch.randn(2,2178)},[2])
        g=ActorCriticEncoder(gobs,{'policy':['policy'],'critic':['critic']},29)
        self.assertTrue(any(isinstance(x,torch.nn.BatchNorm2d) for x in g.map_cnn))
        self.assertEqual(g.act(gobs).shape,(2,29))
        self.assertEqual(g.evaluate(gobs).shape,(2,1))
        self.assertEqual(g.act_inference(gobs)[0].shape,(2,29))
        self.assertTrue(g.load_state_dict(g.state_dict()))
if __name__=='__main__':unittest.main()
