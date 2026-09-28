# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch

#####################################
# Export policy with terrain encoder #
#####################################

def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        
        # ------------------- Auto-detect terrain encoder -------------------
        self.has_terrain_encoder = (
            hasattr(policy, "map_cnn") and 
            hasattr(policy, "mha") and 
            hasattr(policy, "actor_proprio_embedding")
        )
        # Deep-copy terrain encoder modules if available
        if self.has_terrain_encoder:
            self.L = policy.L
            self.W = policy.W
            self.coord_dim = policy.coord_dim
            self.mha_dim = policy.mha_dim
            self.actor_proprio_dim = policy.actor_proprio_dim
            self.map_cnn = copy.deepcopy(policy.map_cnn)
            self.mha = copy.deepcopy(policy.mha)
            self.actor_proprio_embedding = copy.deepcopy(policy.actor_proprio_embedding)
            self.attach_global = getattr(policy, "attach_global", False)
            if self.attach_global:
                self.global_encoder = copy.deepcopy(policy.global_encoder)
                self.query_projector = copy.deepcopy(policy.query_projector)
            
            # Move modules to CPU
            self.map_cnn.cpu()
            self.mha.cpu()
            self.actor_proprio_embedding.cpu()
            if self.attach_global:
                self.global_encoder.cpu()
                self.query_projector.cpu()
        # --------------------------------------------------------

        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            if self.rnn_type == "lstm":
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_lstm
                self.reset = self.reset_memory
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
                self.reset = self.reset_memory
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def _encode_terrain(self, obs):
        """Terrain encoding path (used only when has_terrain_encoder=True)."""
        if not self.has_terrain_encoder:
            raise RuntimeError("Terrain encoder is not available in this model.")
        
        # Ensure batch dimension exists
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        # Split observation: map scan (tail) and proprioception (head)
        map_scan_size = self.L * self.W * self.coord_dim
        map_scan = obs[:, -map_scan_size:].reshape(-1, self.W, self.L, self.coord_dim)
        proprio_obs = obs[:, :-map_scan_size]

        # CNN encoding
        # permute: [batch, L, W, coord] -> [batch, coord, L, W]
        map_scan = map_scan.permute(0, 3, 1, 2)
        
        cnn_features = self.map_cnn(map_scan)
        # permute back: [batch, coord, L, W] -> [batch, L, W, coord] -> flatten spatial dims to sequence dim
        # flatten(1, 2) handles optional downsampling automatically
        cnn_features = cnn_features.permute(0, 2, 3, 1).flatten(1, 2)

        local_features = cnn_features

        # Proprioceptive embedding
        proprio_embedding = self.actor_proprio_embedding(proprio_obs)

        if self.attach_global:
            global_features = self.global_encoder(local_features)
            global_features_max, _ = torch.max(global_features, dim=1)
            query_input = torch.cat([global_features_max, proprio_embedding], dim=-1)
            proprio_embedding = self.query_projector(query_input)

        proprio_embedding = proprio_embedding.unsqueeze(1)
        
        # MHA
        mha_output, _ = self.mha(query=proprio_embedding, key=local_features, value=local_features)
        mha_output = mha_output.squeeze(1)

        encoded_obs = torch.cat([mha_output, proprio_obs], dim=-1)

        if self.attach_global:
            encoded_obs = torch.cat([global_features_max, encoded_obs], dim=-1)

        # Final encoded feature
        return encoded_obs

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward_gru(self, x):
        x = self.normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        x = self.normalizer(x)
        if self.has_terrain_encoder:
            x = self._encode_terrain(x)
        return self.actor(x)

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        
        # ------------------- Auto-detect terrain encoder -------------------
        self.has_terrain_encoder = (
            hasattr(policy, "map_cnn") and 
            hasattr(policy, "mha") and 
            hasattr(policy, "actor_proprio_embedding")
        )
        if self.has_terrain_encoder:
            self.L = policy.L
            self.W = policy.W
            self.coord_dim = policy.coord_dim
            self.mha_dim = policy.mha_dim
            self.actor_proprio_dim = policy.actor_proprio_dim
            self.map_cnn = copy.deepcopy(policy.map_cnn)
            self.mha = copy.deepcopy(policy.mha)
            self.actor_proprio_embedding = copy.deepcopy(policy.actor_proprio_embedding)
            self.attach_global = getattr(policy, "attach_global", False)
            if self.attach_global:
                self.global_encoder = copy.deepcopy(policy.global_encoder)
                self.query_projector = copy.deepcopy(policy.query_projector)
            
            self.map_cnn.cpu()
            self.mha.cpu()
            self.actor_proprio_embedding.cpu()
            if self.attach_global:
                self.global_encoder.cpu()
                self.query_projector.cpu()
        # --------------------------------------------------------

        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h

    def _encode_terrain(self, obs):
        """Terrain encoding path for ONNX export (aligned with JIT path)."""
        if not self.has_terrain_encoder:
            raise RuntimeError("Terrain encoder is not available in this model.")
            
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        map_scan_size = self.L * self.W * self.coord_dim
        map_scan = obs[:, -map_scan_size:].reshape(-1, self.W, self.L, self.coord_dim)
        proprio_obs = obs[:, :-map_scan_size]

        # CNN Features
        map_scan = map_scan.permute(0, 3, 1, 2)
        cnn_features = self.map_cnn(map_scan)
        # Flatten spatial dims into sequence dim (handles downsampling automatically)
        cnn_features = cnn_features.permute(0, 2, 3, 1).flatten(1, 2)
        local_features = cnn_features

        # Proprio Embedding & MHA
        proprio_embedding = self.actor_proprio_embedding(proprio_obs)
        if self.attach_global:
            global_features = self.global_encoder(local_features)
            global_features_max, _ = torch.max(global_features, dim=1)
            query_input = torch.cat([global_features_max, proprio_embedding], dim=-1)
            proprio_embedding = self.query_projector(query_input)
        proprio_embedding = proprio_embedding.unsqueeze(1)
        mha_output, _ = self.mha(query=proprio_embedding, key=local_features, value=local_features)
        mha_output = mha_output.squeeze(1)

        encoded_obs = torch.cat([mha_output, proprio_obs], dim=-1)

        if self.attach_global:
            encoded_obs = torch.cat([global_features_max, encoded_obs], dim=-1)

        return encoded_obs

    def forward(self, x):
        x = self.normalizer(x)
        if self.has_terrain_encoder:
            x = self._encode_terrain(x)
        return self.actor(x)

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=13,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=13,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            # Non-recurrent case: compute the correct input shape
            if self.has_terrain_encoder:
                # Total input dim = proprioceptive dim + map scan dim
                input_dim = self.actor_proprio_dim + (self.L * self.W * self.coord_dim)
                obs = torch.zeros(1, input_dim)
            else:
                obs = torch.zeros(1, self.actor[0].in_features)
                
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=13,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
