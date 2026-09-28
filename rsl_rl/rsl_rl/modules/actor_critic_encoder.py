from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


class ActorCriticEncoder(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # Terrain encoder parameters
        map_scan_dim=(33, 21, 3),  # L=33, W=21, 3D coordinates
        # map_scan_dim=(17, 11, 3),  # L=17, W=11, 3D coordinates
        mha_dim=64,  # MHA feature dimension
        num_heads=16,  # Number of attention heads
        cnn_downsample=True,
        attach_global=False,  # Add max-pooled global feature to query and policy input
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticEncoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.map_scan_dim = map_scan_dim
        self.mha_dim = mha_dim
        self.num_heads = num_heads
        self.L, self.W, self.coord_dim = map_scan_dim
        self.cnn_downsample = cnn_downsample
        self.attach_global = attach_global

        # Option A: concatenate coordinates to CNN features
        # self.cnn_output_dim = mha_dim - 3  # d-3
        # Option B (current): use pure CNN features as MHA input (no coord concat)
        self.cnn_output_dim = mha_dim

        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        map_scan_size = self.L * self.W * self.coord_dim
        actor_proprio_dim = num_actor_obs - map_scan_size
        critic_proprio_dim = num_critic_obs - map_scan_size
        if actor_proprio_dim <= 0 or critic_proprio_dim <= 0:
            raise ValueError(
                f"proprio_dim incorrect, actor_proprio_dim={actor_proprio_dim}, critic_proprio_dim={critic_proprio_dim}"
                f"actor={num_actor_obs}, critic={num_critic_obs}, map_scan={map_scan_size}"
            )

        self.actor_proprio_dim = actor_proprio_dim
        self.critic_proprio_dim = critic_proprio_dim

        self._build_terrain_encoder(self.actor_proprio_dim, self.critic_proprio_dim, self.attach_global)

        actor_input_dim = mha_dim + self.actor_proprio_dim
        critic_input_dim = mha_dim + self.critic_proprio_dim
        if self.attach_global:
            actor_input_dim += mha_dim
            critic_input_dim += mha_dim
            print(f"attach_global=True: Adding global maxpool feature (dim={mha_dim}) to encoded_obs")

        self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.actor_proprio_dim)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP input_dim={actor_input_dim}: {self.actor}")

        self.critic = MLP(critic_input_dim, 1, critic_hidden_dims, activation)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self.critic_proprio_dim)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        print(f"Critic MLP: {self.critic}")
        print(f"Encoder CNN: {self.map_cnn}")
        if self.attach_global:
            print(f"Encoder Query Projector: {self.query_projector}")
            print(f"Encoder Global MLP: {self.global_encoder}")
        print(f"Encoder MHA: {self.mha}")
        print(f"Encoder Actor Proprio: {self.actor_proprio_embedding}")
        print(f"Encoder Critic Proprio: {self.critic_proprio_embedding}")

        # Action noise (log mode guarantees positive std)
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def _build_terrain_encoder(self, actor_proprio_dim, critic_proprio_dim, attach_global):
        """Build terrain encoder modules shared by actor and critic."""
        if not self.cnn_downsample:
            self.map_cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, self.cnn_output_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(self.cnn_output_dim),
            )
        else:
            self.map_cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, self.cnn_output_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(self.cnn_output_dim),
            )

        self.actor_proprio_embedding = nn.Linear(actor_proprio_dim, self.mha_dim)
        self.critic_proprio_embedding = nn.Linear(critic_proprio_dim, self.mha_dim)

        if attach_global:
            self.global_encoder = MLP(self.mha_dim, self.mha_dim, [256, 128], "elu")
            self.query_projector = nn.Linear(self.mha_dim * 2, self.mha_dim)
        else:
            self.global_encoder = None
            self.query_projector = None

        self.mha = nn.MultiheadAttention(embed_dim=self.mha_dim, num_heads=self.num_heads, batch_first=True)
        print(
            f"Terrain Encoder: CNN output dim={self.cnn_output_dim}, "
            f"MHA dim={self.mha_dim}, heads={self.num_heads}"
        )

    def _encode_terrain(self, obs):
        """Encode terrain/map observations into attention-ready features."""
        # Extract map scan from the tail of observation.
        # Stored order and reshape order differ, so swap W/L in reshape to keep spatial alignment.
        map_scan = obs[:, -self.L * self.W * self.coord_dim:].reshape(-1, self.W, self.L, self.coord_dim)

        # Height-only variant
        # height_map = map_scan[:, :, :, 2:3]  # [batch, W, L, 1]
        # Full 3D-coordinate variant (current)
        height_map = map_scan

        height_map = height_map.permute(0, 3, 1, 2)
        cnn_features = self.map_cnn(height_map)
        if not self.cnn_downsample:
            cnn_features = cnn_features.permute(0, 2, 3, 1).reshape(-1, self.L * self.W, self.cnn_output_dim)
        else:
            cnn_features = cnn_features.permute(0, 2, 3, 1).reshape(-1, (self.L // 2 + 1) * (self.W // 2 + 1), self.cnn_output_dim)

        # # Optional branch: concatenate sampled 3D coordinates with CNN features
        # if not self.cnn_downsample:
        #     coords = map_scan.reshape(-1, self.L * self.W, self.coord_dim)
        #     local_features = torch.cat([cnn_features, coords], dim=-1)  # [batch, L*W, mha_dim]
        # else:
        #     # Downsample coordinates to match downsampled CNN grid
        #     coords = map_scan[:, ::2, ::2, :].reshape(-1, (self.L // 2 + 1) * (self.W // 2 + 1), self.coord_dim)
        #     local_features = torch.cat([cnn_features, coords], dim=-1)  # [batch, L'*W', mha_dim]
        local_features = cnn_features  # Current setting: use CNN features only as local features
        
        # Extract proprioceptive features (all non-map terms)
        proprio_obs = obs[:, :-self.L * self.W * self.coord_dim]
        if proprio_obs.shape[1] == self.actor_proprio_dim:
            proprio_embedding = self.actor_proprio_embedding(proprio_obs)
        elif proprio_obs.shape[1] == self.critic_proprio_dim:
            proprio_embedding = self.critic_proprio_embedding(proprio_obs)
        else:
            raise ValueError(
                f"proprio_obs dimension {proprio_obs.shape[1]} does not match actor_proprio_dim {self.actor_proprio_dim} "
                f"or critic_proprio_dim {self.critic_proprio_dim}"
            )

        if self.attach_global:
            global_features = self.global_encoder(local_features)
            global_features_max, _ = torch.max(global_features, dim=1)
            query_input = torch.cat([global_features_max, proprio_embedding], dim=-1)
            proprio_embedding = self.query_projector(query_input)

        proprio_embedding = proprio_embedding.unsqueeze(1)
        mha_output, attention_weights = self.mha(
            query=proprio_embedding,
            key=local_features,
            value=local_features,
        )

        mha_output = mha_output.squeeze(1)
        encoded_obs = torch.cat([mha_output, proprio_obs], dim=-1)
        if self.attach_global:
            encoded_obs = torch.cat([global_features_max, encoded_obs], dim=-1)

        if torch.isnan(encoded_obs).any() or torch.isinf(encoded_obs).any():
            print(f"Warning: encoded_obs contains NaN or Inf: {encoded_obs}")

        return encoded_obs, attention_weights

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print(f"Warning: obs contains NaN or Inf: {obs}")

        encoded_obs, _ = self._encode_terrain(obs)
        mean = self.actor(encoded_obs)

        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print(f"Warning: mean contains NaN or Inf: {mean}")

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        encoded_obs, attention_weights = self._encode_terrain(actor_obs)
        return self.actor(encoded_obs), attention_weights

    def evaluate(self, obs, **kwargs):
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        encoded_obs, _ = self._encode_terrain(critic_obs)
        value = self.critic(encoded_obs)
        if torch.isnan(value).any() or torch.isinf(value).any():
            print(f"Warning: critic value contains NaN or Inf, {value}")
        return value

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            group_data = obs[obs_group]
            if torch.isnan(group_data).any() or torch.isinf(group_data).any():
                print(f"Warning: obs_group '{obs_group}' contains NaN or Inf: {group_data}")
            obs_list.append(group_data)
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            group_data = obs[obs_group]
            if torch.isnan(group_data).any() or torch.isinf(group_data).any():
                print(f"Warning: obs_group '{obs_group}' contains NaN or Inf: {group_data}")
            obs_list.append(group_data)
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
