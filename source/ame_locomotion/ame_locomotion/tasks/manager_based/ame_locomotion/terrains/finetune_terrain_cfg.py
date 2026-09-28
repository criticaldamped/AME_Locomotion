# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

# import isaaclab.terrains as terrain_gen
import ame_locomotion.tasks.manager_based.ame_locomotion.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .loco_hf_terrains_cfg import *

FINETUNE_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=50.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stakes1": terrain_gen.HfDoubleColumnStakesTerrainCfg(
            proportion=0.1, stake_height_max=0.03, stake_side_range=(0.20, 0.40), stake_gap_range=(0.1, 0.3),
            column_gap_range=(0.1, 0.1), column_jitter=0.0, holes_depth=-2.0, platform_width=2.0, border_width=0.25
        ),
        "stakes2": terrain_gen.HfAlternateColumnStakesTerrainCfg(
            proportion=0.2, stake_height_max=0.03, stake_side_range=(0.20, 0.40), stake_gap_range=(0.05, 0.15),
            column_gap_range=(0.0, 0.2), column_jitter=0.0, holes_depth=-2.0, platform_width=2.0, border_width=0.25
        ),
        "stakes3": terrain_gen.HfAlternateColumnStakesTerrainCfg(
            proportion=0.2, stake_height_max=0.03, stake_side_range=(0.20, 0.40), stake_gap_range=(0.05, 0.25),
            column_gap_range=(0.3, 0.2), column_jitter=0.0, holes_depth=-2.0, platform_width=2.0, border_width=0.25
        ),
        "hf_gaps": terrain_gen.HfConcentricGapTerrainCfg(
            proportion=0.1, gap_width_range=(0.2, 0.6), platform_width=2.0, border_width=0.25, gap_depth=-2.0,
            ground_width_range=(0.5, 0.5), ground_height_max=0.03
        ),
        "stonebridge": terrain_gen.HfStonesBridgeTerrainCfg(
            proportion=0.1, platform_width=2.0, border_width=0.25, holes_depth=-2.0,
            stone_height_max=0.03, stone_width_range=(0.25, 0.35), stone_distance_range=(0.3, 0.5),
            stone_length_range=(0.6, 1.0), stone_lateral_distance_range=(0.0, 0.0)
        ),
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.1, rail_height_range=(0.25, 0.05), rail_thickness_range=(0.1, 0.3), platform_width=2.0
        ),
    },
)
"""Rough terrains configuration."""
