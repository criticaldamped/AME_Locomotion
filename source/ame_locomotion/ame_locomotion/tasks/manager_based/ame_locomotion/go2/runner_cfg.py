from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
@configclass
class Go2RunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env=24
    max_iterations=3000
    save_interval=100
    experiment_name='go2_ame'
    obs_groups={'policy':['policy'],'critic':['critic']}
    clip_actions=None
    policy=RslRlPpoActorCriticCfg(class_name='Go2ActorCritic',init_noise_std=0.5,noise_std_type='log',actor_obs_normalization=False,critic_obs_normalization=False,actor_hidden_dims=[256,128,64],critic_hidden_dims=[256,128,64],activation='elu')
    algorithm=RslRlPpoAlgorithmCfg(class_name='CheckedPPO',value_loss_coef=1.,use_clipped_value_loss=True,clip_param=0.2,entropy_coef=0.003,num_learning_epochs=4,num_mini_batches=4,learning_rate=3e-4,schedule='adaptive',gamma=0.99,lam=0.95,desired_kl=0.01,max_grad_norm=1.)
