"""AME Go2 variant: per-sample GroupNorm; strict numerics; no G1 checkpoint changes."""
import torch
from torch import nn
from .actor_critic_encoder import ActorCriticEncoder

def finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'{name}: non-finite value, shape={tuple(value.shape)}')
    return value

class Go2ActorCritic(ActorCriticEncoder):
    def __init__(self, *args, **kwargs):
        if kwargs.pop('state_dependent_std', False):
            raise ValueError('State-dependent std is unsupported by the Go2 v1 contract')
        if kwargs.get('noise_std_type', 'log') != 'log':
            raise ValueError('Go2 requires positive log standard deviation')
        if kwargs.get('attach_global', False) or tuple(kwargs.get('map_scan_dim', (33,21,3))) != (33,21,3):
            raise ValueError('Go2 v1 requires 33x21x3 AME map without global branch')
        if kwargs.get('actor_obs_normalization') or kwargs.get('critic_obs_normalization'):
            raise ValueError('Go2 v1 uses fixed physical scaling; empirical normalization unsupported')
        super().__init__(*args, **kwargs)
        if self.actor_proprio_dim != 45 or self.critic_proprio_dim != 48:
            raise ValueError('Go2 observation contract: policy 45+2079, critic 48+2079')
        for index, layer in enumerate(self.map_cnn):
            if isinstance(layer, nn.BatchNorm2d):
                self.map_cnn[index] = nn.GroupNorm(4, layer.num_features)
        # Small initial mean avoids large random targets from untrained encoder.
        with torch.no_grad():
            self.actor[-1].weight.mul_(0.01)
            self.actor[-1].bias.zero_()
    def _encode_terrain(self, obs):
        finite('AME input', obs)
        encoded, weights = super()._encode_terrain(obs)
        return finite('AME encoding', encoded), weights
    def update_distribution(self, obs):
        finite('log_std', self.log_std)
        std = self.log_std.exp()
        if (std < 0.02).any() or (std > 2.0).any():
            raise FloatingPointError(f'Policy std outside [0.02,2.0]: {std.detach().cpu().tolist()}')
        super().update_distribution(obs)
        finite('action mean', self.action_mean)
    def act(self, obs, **kwargs):
        return finite('sampled action', super().act(obs, **kwargs))
    def act_inference(self, obs):
        actions, _ = super().act_inference(obs)
        return finite('deterministic action', actions)
    def evaluate(self, obs, **kwargs):
        return finite('critic value', super().evaluate(obs, **kwargs))
    def get_actions_log_prob(self, actions):
        finite('log_prob input actions', actions)
        return finite('action log_prob', super().get_actions_log_prob(actions))
