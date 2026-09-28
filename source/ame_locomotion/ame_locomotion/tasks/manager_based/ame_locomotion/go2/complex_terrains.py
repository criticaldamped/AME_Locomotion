"""Go2-sized AME routes, adapted from this project's G1 height-field patterns.

Unlike the G1 functions, every random choice uses a tile-local seeded generator,
and pit depth is a real parameter. Terrain families and difficulty progression
follow the same rough-foundation -> sparse-support refinement design.
"""
import numpy as np
import isaaclab.terrains as terrain
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


def dimensions(difficulty, cfg):
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError('difficulty must be in [0,1]')
    def increasing(pair):
        return pair[0] + difficulty * (pair[1] - pair[0])
    def decreasing(pair):
        return pair[1] - difficulty * (pair[1] - pair[0])
    return dict(support=decreasing(cfg.support_range), gap=increasing(cfg.gap_range),
                width=decreasing(cfg.width_range), separation=increasing(cfg.separation_range))


def route_height_field(difficulty, cfg):
    """x/y in metres; bridge support is longitudinal, width is transverse."""
    params = dimensions(difficulty, cfg)
    if cfg.pit_depth >= 0 or min(params.values()) < 0:
        raise ValueError('Negative pit depth and nonnegative geometry are required')
    scale = cfg.horizontal_scale
    nx, ny = [int(v / scale) for v in cfg.size]
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, round(difficulty * 1_000_000)]))
    heights = np.full((nx, ny), round(cfg.pit_depth / cfg.vertical_scale), dtype=np.int16)
    cx, cy = nx // 2, ny // 2
    side = max(2, round(params['support'] / scale))
    gap = max(1, round(params['gap'] / scale))
    width = max(2, round(params['width'] / scale))
    separation = max(0, round(params['separation'] / scale))
    variation = round(cfg.height_variation * difficulty / cfg.vertical_scale)
    def paint(x, y, sx, sy):
        height = int(rng.integers(-variation, variation + 1))
        x0, y0 = x - sx // 2, y - sy // 2
        heights[max(0,x0):min(nx,x0+sx), max(0,y0):min(ny,y0+sy)] = height
    if cfg.kind == 'gaps':
        # Outward concentric bands, with a guaranteed intact central platform.
        ix, iy = np.meshgrid(np.arange(nx)-cx, np.arange(ny)-cy, indexing='ij')
        radius = np.maximum(np.abs(ix), np.abs(iy))
        platform = round(cfg.platform_width / (2*scale))
        land = max(2, round(params['support'] / scale))
        on_land = (radius <= platform) | (((radius-platform-1) % (gap+land)) >= gap)
        heights[on_land] = 0
    elif cfg.kind in ('double_stakes', 'alternate_stakes', 'bridge'):
        for axis in (0, 1):
            for index, pos in enumerate(range(0, nx if axis == 0 else ny, side+gap)):
                if cfg.kind == 'double_stakes':
                    offsets = (-(side+separation)//2, (side+separation)//2)
                elif cfg.kind == 'alternate_stakes':
                    offsets = ((-1 if index % 2 else 1) * separation//2,)
                else:
                    offsets = (0,)
                for offset in offsets:
                    if axis == 0:
                        paint(pos, cy+offset, side, width if cfg.kind == 'bridge' else side)
                    else:
                        paint(cx+offset, pos, width if cfg.kind == 'bridge' else side, side)
    else:
        raise ValueError(f'Unknown route kind: {cfg.kind}')
    half = round(cfg.platform_width/(2*scale))
    heights[cx-half:cx+half+1, cy-half:cy+half+1] = 0
    return heights

route_mesh = height_field_to_mesh(route_height_field)

@configclass
class Go2RouteTerrainCfg(terrain.HfTerrainBaseCfg):
    function = route_mesh
    kind: str = 'gaps'
    seed: int = 42
    support_range: tuple = (0.25, 0.45)
    gap_range: tuple = (0.05, 0.25)
    width_range: tuple = (0.4, 0.7)
    separation_range: tuple = (0.02, 0.08)
    pit_depth: float = -1.2
    height_variation: float = 0.02
    platform_width: float = 1.0
    border_width: float = 0.1


def stage1_terrain():
    return terrain.TerrainGeneratorCfg(
        seed=42, size=(8.,8.), border_width=10., num_rows=6, num_cols=8,
        horizontal_scale=0.05, vertical_scale=0.005, slope_threshold=0.75,
        curriculum=True, use_cache=False, sub_terrains={
            'stairs_down': terrain.MeshPyramidStairsTerrainCfg(proportion=1., step_height_range=(0.02,0.14),step_width=0.35,platform_width=1.4,border_width=0.5,holes=False),
            'stairs_up': terrain.MeshInvertedPyramidStairsTerrainCfg(proportion=1., step_height_range=(0.02,0.14),step_width=0.35,platform_width=1.4,border_width=0.5,holes=False),
            'slope_down': terrain.HfPyramidSlopedTerrainCfg(proportion=1.,slope_range=(0.,0.25),platform_width=1.4,border_width=0.25),
            'slope_up': terrain.HfInvertedPyramidSlopedTerrainCfg(proportion=1.,slope_range=(0.,0.25),platform_width=1.4,border_width=0.25),
            'boxes': terrain.MeshRandomGridTerrainCfg(proportion=1.,grid_width=0.45,grid_height_range=(0.02,0.10),platform_width=1.4),
            'rough': terrain.HfRandomUniformTerrainCfg(proportion=1.,noise_range=(0.01,0.04),noise_step=0.01,downsampled_scale=0.1,border_width=0.25),
            'stepping_stones': terrain.HfSteppingStonesTerrainCfg(proportion=1.,stone_height_max=0.02,stone_width_range=(0.35,0.55),stone_distance_range=(0.05,0.12),holes_depth=-1.2,platform_width=1.0,border_width=0.1),
            'gaps': Go2RouteTerrainCfg(proportion=1.,kind='gaps',support_range=(0.65,0.85),gap_range=(0.05,0.15),height_variation=0.),
        })


def stage2_terrain():
    g = stage1_terrain()
    g.sub_terrains = {
        'stairs_down': terrain.MeshPyramidStairsTerrainCfg(proportion=1.,step_height_range=(0.04,0.20),step_width=0.3,platform_width=1.4,border_width=0.5,holes=False),
        'stairs_up': terrain.MeshInvertedPyramidStairsTerrainCfg(proportion=1.,step_height_range=(0.04,0.20),step_width=0.3,platform_width=1.4,border_width=0.5,holes=False),
        'double_stakes': Go2RouteTerrainCfg(proportion=1.,kind='double_stakes',support_range=(0.25,0.45),gap_range=(0.05,0.25),separation_range=(0.02,0.08)),
        'alternate_stakes': Go2RouteTerrainCfg(proportion=1.,kind='alternate_stakes',support_range=(0.25,0.45),gap_range=(0.05,0.25),separation_range=(0.15,0.30)),
        'bridge': Go2RouteTerrainCfg(proportion=1.,kind='bridge',support_range=(0.35,0.55),gap_range=(0.05,0.25),width_range=(0.4,0.7)),
        'gaps': Go2RouteTerrainCfg(proportion=1.,kind='gaps',support_range=(0.5,0.8),gap_range=(0.10,0.35),height_variation=0.),
        'rails': terrain.MeshRailsTerrainCfg(proportion=1.,rail_height_range=(0.04,0.16),rail_thickness_range=(0.1,0.2),platform_width=1.4),
        'rough': terrain.HfRandomUniformTerrainCfg(proportion=1.,noise_range=(0.01,0.06),noise_step=0.01,downsampled_scale=0.1,border_width=0.25),
    }
    return g
