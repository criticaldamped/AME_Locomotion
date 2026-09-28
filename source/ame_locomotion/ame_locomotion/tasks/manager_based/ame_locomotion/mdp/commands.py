from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp import TerrainBasedPose2dCommandCfg, TerrainBasedPose2dCommand

from isaaclab.utils import configclass


class TimeLimitedTerrainBasedPose2dCommand(TerrainBasedPose2dCommand):
    """Command generator that generates pose commands containing a 3-D position, heading, and time-to-go.

    The command generator samples uniform 2D positions around the environment origin. It sets
    the height of the position command to the default root height of the robot. The heading
    command is either set to point towards the target or is sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.

    The command tensor includes the time remaining to reach the target as the last element.
    Shape: (num_envs, 5) -> [x, y, z, heading, time_to_go]
    """

    cfg: TimeLimitedTerrainBasedPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TimeLimitedTerrainBasedPose2dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose and time-to-go in base frame. Shape is (num_envs, 5).

        Indices:
            0-2: Position command (x, y, z) in base frame.
            3: Heading command in base frame.
            4: Time to go.
        """
        return torch.cat([
            self.pos_command_b,
            self.heading_command_b.unsqueeze(1),
            self.time_left.unsqueeze(1)
        ], dim=1)


@configclass
class TimeLimitedTerrainBasedPose2dCommandCfg(TerrainBasedPose2dCommandCfg):

    class_type = TimeLimitedTerrainBasedPose2dCommand
