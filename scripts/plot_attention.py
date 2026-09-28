import numpy as np
import matplotlib.pyplot as plt
import os
from rsl_rl.utils import PROJ_ROOT_DIR

# Load attention weights
attn_weights = np.load(os.path.join(PROJ_ROOT_DIR, 'attention_weights.npy'))  # (200, 1, 1, 187)
steps = attn_weights.shape[0]

# Map size: 17x11
map_h, map_w = 17, 11

# Create output folder
save_dir = os.path.join(PROJ_ROOT_DIR, 'attn_vis')
os.makedirs(save_dir, exist_ok=True)

# # Compute global min/max before the loop
# vmin = attn_weights.min()
# vmax = attn_weights.max()

# Support both single-head and multi-head shapes
if len(attn_weights.shape) == 5:       # Multi-head shape: (steps, batch, num_heads, tgt_len, src_len)
    num_heads = attn_weights.shape[2]
elif len(attn_weights.shape) == 4:     # Averaged shape: (steps, batch, tgt_len, src_len)
    num_heads = 1
    attn_weights = np.expand_dims(attn_weights, axis=2)
else:
    raise ValueError(f"Unknown attention_weights shape: {attn_weights.shape}")

import math
cols = min(4, num_heads)
rows = math.ceil(num_heads / cols)

for t in range(steps):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if num_heads == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
        
    fig.suptitle(f'Attention Weights Step {t}', fontsize=16)

    for h in range(num_heads):
        r, c = h // cols, h % cols
        ax = axes[r, c]
        
        # Extract env-0 data for head h
        vec = attn_weights[t, 0, h, 0][::-1]   # Reverse to match map orientation
        attn_map = vec.reshape(map_h, map_w, order='F')  # Column-major reshape
        
        im = ax.imshow(
            attn_map, 
            cmap='viridis', 
            interpolation='nearest', 
            extent=[0.5, -0.5, -0.8, 0.8], 
            # vmin=vmin, 
            # vmax=vmax
        )
        ax.set_title(f'Head {h}')
        # Move y-axis labels to the right
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        # Set axis ticks (use 0.2 step to reduce overlap in multi-head plots)
        ax.set_xticks(np.arange(0.5, -0.6, -0.2))  
        ax.set_yticks(np.arange(-0.8, 0.9, -0.2) if -0.8 > 0.9 else np.arange(-0.8, 0.9, 0.2))
        fig.colorbar(im, ax=ax, location='left', fraction=0.046, pad=0.04)
        
    # Hide extra subplot areas
    for h in range(num_heads, rows * cols):
        r, c = h // cols, h % cols
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/attn_step_{t:03d}.png')
    plt.close()

# Optional: generate animation
import imageio
images = [imageio.imread(f'{save_dir}/attn_step_{t:03d}.png') for t in range(steps)]
imageio.mimsave(f'{save_dir}/attn_weights.gif', images, duration=0.01667)