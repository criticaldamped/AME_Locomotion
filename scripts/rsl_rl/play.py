# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import math

from isaaclab.app import AppLauncher
from rsl_rl.utils import PROJ_ROOT_DIR

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--save_attention_weights", action="store_true", default=False, help="Save attention weights during play.")
parser.add_argument("--vis_attention", action="store_true", default=False, help="Visualize attention weights as colored markers (env 0).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=300, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="AME-G1-29DOF-Play-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--manual_headless", action="store_true", default=False, help=f"Manual Headless mode for convenient debug in IDE. \
                                    Available only when app_launcher has been modified, Otherwise, there is no effect for this parameter.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
args_cli.enable_cameras = True
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from typing import Optional

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from exporter import export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import ame_locomotion.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    attention_weights_list = []
    attention_visualizer = None
    identity_quat = None
    attention_num_bins = 8  # Number of discrete color bins

    def _infer_scan_grid_shape() -> tuple[int, int] | None:
        """Infer the 2D scan grid shape (rows, cols) from encoder/env config."""
        # Prefer explicit encoder grid dimensions
        if hasattr(policy_nn, "W") and hasattr(policy_nn, "L"):
            rows = int(getattr(policy_nn, "W"))
            cols = int(getattr(policy_nn, "L"))
            if rows > 0 and cols > 0:
                return rows, cols

        # Fallback to environment scanner pattern config
        try:
            pattern_cfg = env_cfg.scene.height_scanner.pattern_cfg
            resolution = float(pattern_cfg.resolution)
            size_x, size_y = pattern_cfg.size
            # GridPattern includes both endpoints, so use +1
            cols = int(round(float(size_x) / resolution)) + 1
            rows = int(round(float(size_y) / resolution)) + 1
            if rows > 0 and cols > 0:
                return rows, cols
        except Exception:
            return None

        return None

    scan_grid_shape = _infer_scan_grid_shape()

    # --- attention viz helpers ---
    def _build_attention_visualizer(device):
        nonlocal identity_quat
        # Build a discretized blue->red color palette for attention
        colors = []
        for i in range(attention_num_bins):
            t = i / max(1, attention_num_bins - 1)  # 0~1
            colors.append((t, 0.0, 1.0 - t))  # blue -> red
        markers = {
            f"dot_{i}": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors[i]),
            )
            for i in range(attention_num_bins)
        }
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/attention",
            markers=markers,
        )
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        return VisualizationMarkers(cfg)

    def _vis_attention_on_terrain(attn: torch.Tensor, env):
        """Visualize attention weights using color and marker size."""
        if attn is None or attn.numel() == 0:
            return
        
        try:
            height_scanner = env.unwrapped.scene["height_scanner"]
            if not hasattr(height_scanner.data, "ray_hits_w"):
                print("[WARN] height_scanner.data.ray_hits_w not available")
                return

            # Use ray hit points from env 0
            ray_hits = height_scanner.data.ray_hits_w[0]  # [num_rays, 3]

            # Attention weights from env 0 [N_attn]
            attn0 = attn[0, 0]
            n_attn = int(attn0.shape[0])
            n_rays = int(ray_hits.shape[0])

            # Dynamically align ray points and attention length:
            # 1) same length: one-to-one mapping
            # 2) more rays: prefer grid downsampling, otherwise uniform sampling
            # 3) more attention entries: truncate to ray count
            if n_rays == n_attn:
                selected_hits = ray_hits
                attn_vals = attn0
            elif n_rays > n_attn:
                selected_hits = None
                if scan_grid_shape is not None:
                    rows, cols = scan_grid_shape
                    if rows * cols == n_rays:
                        ray_hits_grid = ray_hits.reshape(rows, cols, 3)
                        if hasattr(policy_nn, "cnn_downsample") and bool(getattr(policy_nn, "cnn_downsample")):
                            ds_hits = ray_hits_grid[::2, ::2, :].reshape(-1, 3)
                            if ds_hits.shape[0] == n_attn:
                                selected_hits = ds_hits
                if selected_hits is None:
                    sample_idx = torch.linspace(0, n_rays - 1, steps=n_attn, device=ray_hits.device).round().long()
                    selected_hits = ray_hits[sample_idx]
                attn_vals = attn0
            else:
                if timestep % 50 == 0:
                    print(
                        f"[WARN] Attention length ({n_attn}) is larger than ray count ({n_rays}); "
                        "this is unexpected and attention will be truncated."
                    )
                selected_hits = ray_hits
                attn_vals = attn0[:n_rays]

            # Filter invalid rays (abnormal z typically means no hit)
            valid_mask = (selected_hits[:, 2] > -50.0) & (selected_hits[:, 2] < 100.0)
            if not valid_mask.any():
                return

            selected_hits = selected_hits[valid_mask]
            attn_vals     = attn_vals[valid_mask]

            # Normalize weights to [0, 1]
            attn_norm = (attn_vals - attn_vals.min()) / (attn_vals.max() - attn_vals.min() + 1e-6)
            n_vis = selected_hits.shape[0]

            # Color bins (blue -> red)
            bins = (attn_norm * (attention_num_bins - 1)).long().clamp(0, attention_num_bins - 1)

            # Slight z offset to avoid embedding into the ground
            points = selected_hits.clone()
            points[:, 2] += 0.05

            # Marker size scales with attention weight (0.5x ~ 3.0x)
            scale_factors = attn_norm * (3.0 - 0.5) + 0.5
            scales = scale_factors.unsqueeze(1).expand(-1, 3)  # [n_vis, 3]

            if timestep % 50 == 0:
                print(f"[DEBUG] Attn viz: valid={n_vis}, "
                      f"attn=[{attn_vals.min():.4f}, {attn_vals.max():.4f}], "
                      f"scale=[{scale_factors.min():.2f}x, {scale_factors.max():.2f}x]")

            orientations = identity_quat.unsqueeze(0).expand(n_vis, -1)
            attention_visualizer.visualize(points, orientations, marker_indices=bins, scales=scales)

        except Exception as e:
            print(f"[WARN] Attention terrain viz failed: {e}")
            import traceback
            traceback.print_exc()

    if args_cli.save_attention_weights:
        if args_cli.vis_attention:
            attention_visualizer = _build_attention_visualizer(env.unwrapped.device)
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions, attention_weights, *_ = policy(obs)
                attention_weights_list.append(attention_weights.cpu().numpy())
                if args_cli.vis_attention and attention_visualizer is not None:
                    _vis_attention_on_terrain(attention_weights, env)
                # env stepping
                obs, *_ = env.step(actions)

                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break
            
            # time delay for real-time playback
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    else:
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions, *_ = policy(obs)
                # env stepping
                obs, *_ = env.step(actions)
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break
            
            # time delay for real-time playback
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    # Save attention weights after simulation
    if len(attention_weights_list) > 0:
        import numpy as np
        np.save(os.path.join(PROJ_ROOT_DIR, 'attention_weights.npy'), np.array(attention_weights_list))
        print(f"[INFO] Attention weights saved to attention_weights.npy, shape: {np.array(attention_weights_list).shape}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
