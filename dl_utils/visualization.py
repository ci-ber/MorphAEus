import matplotlib.pyplot as plt
import numpy as np
import torch
import umap.umap_ as umap
import seaborn as sns
import logging
import wandb

def plot_warped_grid(ax, disp, bg_img=None, interval=3, title="$\mathcal{T}_\phi$", fontsize=30, color=(0.85, 0.27, 0.41, 0.75), linewidth=1.0):
    """
    Visualization of the deformation grids
    disp shape (2, H, W)
            code from: https://github.com/qiuhuaqi/midir
    """
    if bg_img is not None:
        background = bg_img
    else:
        background = np.zeros(disp.shape[1:])

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(0, background.shape[1] - 1, interval),
                                       indexing='ij')

    new_grid_H = id_grid_H + disp[0, id_grid_H, id_grid_W]
    new_grid_W = id_grid_W + disp[1, id_grid_H, id_grid_W]

    kwargs = {"linewidth": linewidth, "color": color}
    # matplotlib.plot() uses CV x-y indexing
    for i in range(new_grid_H.shape[0]):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **kwargs)  # each draws a horizontal line
    for i in range(new_grid_H.shape[1]):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **kwargs)  # each draws a vertical line

    ax.set_title(title, fontsize=fontsize)
    ax.imshow(background, cmap='gray')
    # ax.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    return ax
